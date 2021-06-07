"""
Author: Fei Ding (feid@clemson.edu)
Date: Jul 27, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp_head import MLPHead
from .memory import ContrastMemory

eps=1e-7

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, opt, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        #self.embeds = Embed(opt.t_dim, opt.t_dim//2)
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        #self.embed_s = Head(opt.s_dim, opt.feat_dim) 

    def forward(self, feat_s, feat_t, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if feat_s.is_cuda
                  else torch.device('cpu'))

        #if len(features.shape) < 3:
        #    raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                     'at least 3 dimensions are required')
        #if len(features.shape) > 3:
        #    features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = feat_s.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            #mask = torch.eq(labels, labels.T).float().to(device)
            mask = torch.eq(labels, torch.transpose(labels,0,1)).float().to(device)
        else:
            mask = mask.float().to(device)
        
        embed_s = self.embed_s(feat_s)
        embed_t = self.embed_t(feat_t)
        anchor_feature = torch.cat([embed_s, embed_t], dim=0)
        anchor_count = 2

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(anchor_feature,0,1)),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, anchor_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class AlignLoss(torch.nn.Module):
    def __init__(self, opt, p=2):
        super(AlignLoss, self).__init__()
        self.p = p
        self.mlp = MLPHead(in_channels=opt.s_dim, \
                mlp_hidden_size=16*opt.t_dim, projection_size=opt.t_dim)
 
    def forward(self, feat_s, feat_t):
        feat_s = self.mlp(feat_s)
        feat_s = F.normalize(feat_s, dim=-1, p=self.p)
        feat_t = F.normalize(feat_t, dim=-1, p=self.p)
        return 2 - 2 * (feat_s * feat_t).sum(dim=-1).mean()


class CorrLoss(torch.nn.Module):
    def __init__(self, opt, temperature=0.5):
        super(CorrLoss, self).__init__()
        self.temperature = temperature
        self.mlp = MLPHead(in_channels=opt.s_dim, \
                mlp_hidden_size=opt.s_dim, projection_size=opt.s_dim)

    def forward(self, feat_s, feat_t):
        batch_size = int(feat_s.size(0) / 4)
        nor_index = (torch.arange(4*batch_size) % 4 == 0).cuda()
        aug_index = (torch.arange(4*batch_size) % 4 != 0).cuda()

        f_s = self.mlp(feat_s)
        f_s_nor = f_s[nor_index]
        f_s_aug = f_s[aug_index]
        f_s_nor = f_s_nor.unsqueeze(2).expand(-1,-1,3*batch_size).transpose(0,2)
        f_s_aug = f_s_aug.unsqueeze(2).expand(-1,-1,1*batch_size)
        s_simi = F.cosine_similarity(f_s_aug, f_s_nor, dim=1)

        f_t_nor = feat_t[nor_index]
        f_t_aug = feat_t[aug_index]
        f_t_nor = f_t_nor.unsqueeze(2).expand(-1,-1,3*batch_size).transpose(0,2)
        f_t_aug = f_t_aug.unsqueeze(2).expand(-1,-1,1*batch_size)
        t_simi = F.cosine_similarity(f_t_aug, f_t_nor, dim=1)
        t_simi = t_simi.detach()

        s_simi_log = F.log_softmax(s_simi / self.temperature, dim=1)
        t_simi_log = F.softmax(t_simi / self.temperature, dim=1)
        loss_div = F.kl_div(s_simi_log, t_simi_log, reduction='batchmean')
        return loss_div

class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Head(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Head, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim_in, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, dim_out)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
