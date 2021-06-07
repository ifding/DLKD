import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from models import model_dict
from helper.losses import SupConLoss, CRDLoss, AlignLoss, CorrLoss
from dataset.imagenet import get_imagenet_dataloader

model_names = sorted(name for name in model_dict
    if name.islower() and not name.startswith("__")
    and callable(model_dict[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='../datasets/Imagenet/',
                    help='path to dataset')
parser.add_argument('-s', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-t', '--arch_teacher', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet34)')
parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for div')
parser.add_argument('-b', '--beta', type=float, default=1, help='weight balance for KD')
parser.add_argument('-d', '--delta', type=float, default=0, help='weight balance for reg')

parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')



best_acc1 = 0
#args, unparsed = parser.parse_known_args()


def main():
    args = parser.parse_args()
    #print(args)
    print_parameters(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    # tensorboard setting
    print(">>>>>>>>>>>>>>>>>MSKD training >>>>>>>>>>>>>>>>")
    exp_name = 'S_{}_T_{}_{}_{}_r_{}_a_{}_b_{}_d_{}'.format(args.arch, args.arch_teacher, 'imagenet', 'mlkd',
                                                                args.gamma, args.alpha, args.beta, args.delta)
    log_dir = 'save/student_tensorboards/' + exp_name + '/'
    
    warnings.filterwarnings("ignore")
    writer = SummaryWriter(log_dir=log_dir)
    print("Tensorboard dir: {}".format(log_dir))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    # create model
    print("=> creating model '{}'".format(args.arch))
    model = model_dict[args.arch](num_classes=1000)
    print("=> creating teacher model '{}'".format(args.arch_teacher))
    teacher_model = model_dict[args.arch_teacher](num_classes=1000)
    teacher_model.load_state_dict(torch.load(args.path_t))
    # Use the following lines if you load a pre-trained model by yourself
    #teacher_file = 'pretrained_model/' + 'teacher_model_'+ str(args.arch)+'.pth'
    #teacher_model = torch.load(teacher_file)

    data = torch.randn(2, 3, 224, 224)
    model.eval()
    teacher_model.eval()
    feat_s, _ = model(data, is_feat=True)
    feat_t, _ = teacher_model(data, is_feat=True)

    args.s_dim = feat_s[-1].shape[1]
    args.t_dim = feat_t[-1].shape[1]

    # DataParallel will divide and allocate batch_size to all available GPUs
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        teacher_model.features = torch.nn.DataParallel(teacher_model.features)
        teacher_model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
        teacher_model = torch.nn.DataParallel(teacher_model).cuda()

    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    module_list = nn.ModuleList([])
    module_list.append(model)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model)

    criterion_cls = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_corr = CorrLoss(args, temperature=0.5)
    criterion_kd = SupConLoss(args, temperature=0.07)
    criterion_align = AlignLoss(args)

    module_list.append(criterion_corr.mlp)
    module_list.append(criterion_kd.embed_s)
    module_list.append(criterion_kd.embed_t)
    module_list.append(criterion_align.mlp)

    trainable_list.append(criterion_corr.mlp)
    trainable_list.append(criterion_kd.embed_s)
    trainable_list.append(criterion_kd.embed_t)
    trainable_list.append(criterion_align.mlp)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_corr)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss
    criterion_list.append(criterion_align)    # Regression loss

    optimizer = torch.optim.SGD(trainable_list.parameters(), args.lr*(args.batch_size/256),
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(teacher_model)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = False


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    train_dataset, val_dataset = get_imagenet_dataloader(batch_size=args.batch_size, 
            num_workers=args.workers)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers//2,
                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # learning rate scheduler, decay 1/10 in 30, 60 and 90 epoch
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    # Test the accuracy of teacher
    acc1, acc5, test_loss = validate(val_loader, teacher_model, criterion_cls, args)
    print(">>>>>>>>>>>>>>>The teacher accuracy, top1:{}, top5:{}>>>>>>>>>>>>".format(acc1, acc5))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # learning rate update
        scheduler.step(epoch)

        # train for one epoch
        train_acc1, train_acc5, train_loss_CE = train(train_loader, module_list, criterion_list, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5, test_loss = validate(val_loader, model, criterion_cls, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            print("'The best acc', epoch:{},  top1:{}, top5:{}".format(epoch+1, acc1, acc5))


        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, dir = log_dir)

        # Tensorboard
        writer.add_scalar('Train_acc_top1', train_acc1, epoch)
        writer.add_scalar('Train_acc_top5', train_acc5, epoch)
        writer.add_scalar('Train_loss_CE', train_loss_CE, epoch)
        writer.add_scalar('Test_acc_top1', acc1, epoch)
        writer.add_scalar('Test_acc_top5', acc5, epoch)
        writer.add_scalar('Test_loss_CE', test_loss, epoch)
            # export scalar data to JSON for external processing
    writer.close()


def train(train_loader, module_list, criterion_list, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_CE = AverageMeter('Loss_CE', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    print('learning rate: {}'.format(optimizer.param_groups[0]['lr']))
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    #model.train()
    #teacher_model.eval()
    
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_corr = criterion_list[1]
    criterion_kd = criterion_list[2]
    criterion_align = criterion_list[3]
   
    model = module_list[0]
    teacher_model = module_list[-1]


    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        c,h,w = images.size()[-3:]
        images = images.view(-1, c, h, w)

        batch_size = int(images.size(0) / 4)
        nor_index = (torch.arange(4*batch_size) % 4 == 0).cuda()
        aug_index = (torch.arange(4*batch_size) % 4 != 0).cuda()

        feat_s, logit_s = model(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = teacher_model(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        loss_cls = criterion_cls(logit_s[nor_index], target)

        f_s = feat_s[-1]
        f_t = feat_t[-1]

        loss_div = criterion_corr(f_s, f_t)
        
        aug_target = target.unsqueeze(1).expand(-1,4).contiguous().view(-1).long().cuda()
        loss_kd = criterion_kd(f_s, f_t, aug_target) 

        # Regression loss
        loss_reg = criterion_align(f_s, f_t)

        loss = args.gamma * loss_cls + args.alpha * loss_div + args.beta * loss_kd + args.delta * loss_reg

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logit_s[nor_index], target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        losses_CE.update(loss_cls.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, top5.avg, losses_CE.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar'):
    dir_name = dir + filename
    torch.save(state, dir_name)
    if is_best:
        shutil.copyfile(dir_name, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def print_parameters(argv):
    for arg in vars(argv):
        print(arg, getattr(argv, arg))

if __name__ == '__main__':
    main()
