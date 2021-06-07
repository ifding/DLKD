"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from explainability.Interpreter import calculate_regularization, Interpreter
import cv2
import numpy as np

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar100 import get_cifar100_dataloaders_augment

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init

from helper.losses import SupConLoss, CRDLoss, REGLoss, DIVLoss


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='ShuffleV2',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='mlkd', choices=['kd', 'mlkd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for div')
    parser.add_argument('-b', '--beta', type=float, default=1, help='weight balance for KD')
    parser.add_argument('-d', '--delta', type=float, default=0, help='weight balance for reg')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()
    
    opt.path_t = './save/models/wrn_40_2_vanilla/wrn_40_2.pth'
    opt.path_s = './save/student_model/S_ShuffleV1_T_wrn_40_2_cifar100_MLKD_r_1.0_a_10.0_b_1.0_d_20.0/ShuffleV1_best.pth'
    opt.exp = 'ShuffleV1'

    opt.model_s = 'ShuffleV1'

    opt.batch_size = 2

    opt.model_t = get_teacher_name(opt.path_t)

    opt.save_folder = os.path.join('./save', 'explain')
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    #model.load_state_dict(torch.load(model_path)['model'])
    model.load_state_dict(torch.load(model_path)['state_dict'])
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader  = get_cifar100_dataloaders_augment(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)
 
    global model_t
    global model_s

    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    model_s.load_state_dict(torch.load(opt.path_s)['model'])
    
    model_t.cuda()
    model_t.eval()

    model_s.cuda()
    model_s.eval()

    # 0, 1, 2, 3
    DEG = 0

    images = []
    labels = []
    for idx, (input, target) in enumerate(val_loader):
        input = input[:,DEG,:,:,:].float()
        input = input.squeeze(0).view(3, -1).transpose(0, 1).cuda()
        target = target.cpu().numpy()
        images.append(input)
        labels.append(target)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pixels = [str(i) for i in range(1024)]
    
    for p in model_t.parameters():
        p.requires_grad = False
    
    for p in model_s.parameters():
        p.requires_grad = False
    
    def Phi(x):
        x = x.transpose(0, 1).view(3, 32, 32)
        x.unsqueeze_(0)

        #feats, _ = model_t(x, is_feat=True)
        feats, _ = model_s(x, is_feat=True)
        
        return feats[-1]

    regularization = calculate_regularization(images, Phi, device=device)

    raw_img = []
    results = []

    for i in range(len(images)):
        img = images[i]
        interpreter = Interpreter(x=img, Phi=Phi, regularization=regularization, scale=10 * 0.1, words=pixels).to(device)
        interpreter.optimize(iteration=5000, lr=0.5, show_progress=True)
        sigma = interpreter.get_sigma()
        sigma = np.expand_dims(sigma.reshape(32, 32), axis=0)
        results.append(sigma)
        
        img = img.transpose(0, 1).view(3, 32, 32)
        img_array = img.cpu().detach().numpy()
        img_array = np.expand_dims(img_array, axis=0)
        raw_img.append(img_array)

        print(i, sigma.shape, img_array.shape)
        #if i >= 1999:
        if i >= 999:
            break
    
    raw_img = np.vstack(raw_img)
    results = np.vstack(results)
    
    print(raw_img.shape)
    print(results.shape)

    np.save(os.path.join(opt.save_folder, '{EXP}_img_{DEG}.npy'.format(EXP=opt.exp, DEG=DEG)), raw_img)
    np.save(os.path.join(opt.save_folder, '{EXP}_res_{DEG}.npy'.format(EXP=opt.exp, DEG=DEG)), results)
    
    #labels = np.vstack(labels)
    #save_file = os.path.join(opt.save_folder, 'lbl.npy')
    #np.save(save_file, labels) 

    #save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))



if __name__ == '__main__':
    main()
