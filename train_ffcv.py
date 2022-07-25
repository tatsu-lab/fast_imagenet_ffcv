import argparse
import os
import random
import shutil
import time
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with FFCV')
parser.add_argument('-t', '--train-file', default='/self/scr-sync/nlp/imagenet_ffcv/train_512_1.0_90.ffcv',
                    help='path to FFCV train dataset')
parser.add_argument('-v', '--val-file', default='/self/scr-sync/nlp/imagenet_ffcv/val_512_1.0_90.ffcv',
                    help='path to FFCV val dataset')
parser.add_argument('-d', '--data-dir', metavar='DIR', default='/self/scr-sync/nlp/imagenet',
                    help='path to dataset (default: /self/scr-sync/nlp/imagenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading processes per gpu')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, 
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp32', action='store_true',
                    help='train in full precision (instead of fp16)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='batch size on each gpu')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='base learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-s', '--save-dir', default='./',
                    help='directory to save checkpoints')
parser.add_argument('--dist-url', default=f'tcp://127.0.0.1:{random.randint(1, 9999)+30000}', type=str,
                    help='url used to set up distributed training')


best_acc1 = 0

def main():
    args = parser.parse_args()
    args.ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.ngpus, args=(args,))


def main_worker(gpu, args):
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))
    dist.init_process_group('nccl', init_method=args.dist_url, rank=args.gpu, world_size=args.ngpus)
    torch.cuda.set_device(args.gpu)
    
    global best_acc1
    cudnn.benchmark = True

    # create model
    if args.pretrained:
        model = models.__dict__[args.arch](weights='IMAGENET1K_V1')
    else:
        model = models.__dict__[args.arch]()
    model = model.to(memory_format=torch.channels_last).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    # linear scale learning rate with 256 base batch size
    args.lr *= args.batch_size * args.ngpus / 256
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # decay lr by 10 every 30 epochs
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # fp16 loss scaler
    scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

    train_image_pipeline = [
        RandomResizedCropRGBImageDecoder((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(f'cuda:{args.gpu}', non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16 if not args.fp32 else np.float32),
    ]

    val_image_pipeline = [
        CenterCropRGBImageDecoder((224, 224), ratio=224/256),
        ToTensor(),
        ToDevice(f'cuda:{args.gpu}', non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16 if not args.fp32 else np.float32)
    ]

    label_pipeline = [IntDecoder(), ToTensor(), Squeeze(),
                      ToDevice(f'cuda:{args.gpu}', non_blocking=True)]

    train_loader = Loader(args.train_file, batch_size=args.batch_size, num_workers=args.workers,
                          order=OrderOption.RANDOM, os_cache=True, drop_last=True,
                          pipelines={'image': train_image_pipeline, 'label': label_pipeline},
                          distributed=True, seed=0)

    val_loader = Loader(args.val_file, batch_size=args.batch_size, num_workers=args.workers,
                        order=OrderOption.SEQUENTIAL, os_cache=True, drop_last=False,
                        pipelines={'image': val_image_pipeline, 'label': label_pipeline},
                        distributed=True)

    if args.evaluate:
        with torch.no_grad():
            validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, scaler, epoch, args)
        with torch.no_grad():
            acc1 = validate(val_loader, model, criterion, args)
        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if args.gpu == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best, args.save_dir)


def train(train_loader, model, criterion, optimizer, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, top1], 
                              prefix="Epoch: [{}]".format(epoch), is_master=args.gpu==0)

    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        # compute loss in fp16 (unless disabled)
        with torch.cuda.amp.autocast(enabled=not args.fp32):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix='Test: ',
                              is_master=args.gpu==0)

    model.eval()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):

        # compute output in fp16 (unless disabled)
        with torch.cuda.amp.autocast(enabled=not args.fp32):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    top1.all_reduce()
    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filedir):
    torch.save(state, os.path.join(filedir, 'checkpoint.pth'))
    if is_best:
        shutil.copyfile(os.path.join(filedir, 'checkpoint.pth'), 
                        os.path.join(filedir, 'checkpoint_best.pth'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count]).cuda()
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", is_master=True):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.is_master = is_master

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.is_master:
            print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        if self.is_master:
            print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
