# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib

from torch import nn, optim
from torchvision import models, datasets, transforms
import torch
import torchvision

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')
parser.add_argument('pretrained', type=Path, metavar='FILE',
                    help='path to pretrained model')
parser.add_argument('data', type=Path, metavar='DIR', default="./datasets/", nargs='?',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str, choices=('cifar10', 'cifar100'),
                    help='cifar dataset (automatically downloaded in data folder)')
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.3, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/lincls/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--encoder', default='resnet50', type=str, choices=('resnet18', 'resnet34', 'resnet50'),
                    help="encoder backbone model")

cifar_mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

cifar_std = {
    'cifar10': (0.2470, 0.2435, 0.2616),
    'cifar100': (0.2675, 0.2565, 0.2761),
}


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    num_classes = 10 if args.dataset == 'cifar10' else 100

    if args.encoder == 'resnet18':
        model = models.resnet18(num_classes = num_classes).cuda(gpu)
    elif args.encoder == 'resnet34':
        model = models.resnet34(num_classes = num_classes).cuda(gpu)
    else:
        model = models.resnet50(num_classes = num_classes).cuda(gpu)

    state_dict = torch.load(args.pretrained, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []

    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    if args.weights == 'freeze':
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        best_val_acc = ckpt['best_val_acc']
        best_test_acc = ckpt['best_test_acc']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0
        best_val_acc = argparse.Namespace(top1=0, top5=0)
        best_test_acc = argparse.Namespace(top1=0, top5=0)

    # DATASET

    normalize = transforms.Normalize(mean=cifar_mean[args.dataset], std=cifar_std[args.dataset])

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])

    test_transform = transforms.Compose([
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
    ])

    if args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10
    else:
        dataset = torchvision.datasets.CIFAR100

    train_dataset = dataset(args.data / 'train', download=True, train=True, transform=train_transform)
    val_dataset = dataset(args.data / 'train', download=True, train=True, transform=test_transform)
    
    # split in train and val
    seed = 10
    num_train = int( args.train_percent * len(train_dataset) / 100 )
    train_idx = list(range(len(train_dataset)))
    random.Random(seed).shuffle(train_idx)
    num_val = int(0.1 * num_train)
    num_train = num_train - num_val
    val_idx = train_idx[num_train:]
    train_idx = train_idx[:num_train]

    val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)

    test_dataset = dataset(args.data / 'test', download=True, train=False, transform=test_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False
        train_sampler.set_epoch(epoch)
        for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            output = model(images.cuda(gpu, non_blocking=True))
            loss = criterion(output, target.cuda(gpu, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_classifier = pg[0]['lr']
                    lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
                    stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                                 lr_classifier=lr_classifier, loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        # evaluate
        model.eval()
        if args.rank == 0:
            # valdation
            val_top1 = AverageMeter('ValAcc@1')
            val_top5 = AverageMeter('ValAcc@5')
            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 5))
                    val_top1.update(acc1[0].item(), images.size(0))
                    val_top5.update(acc5[0].item(), images.size(0))
            
            #test
            test_top1 = AverageMeter('TestAcc@1')
            test_top5 = AverageMeter('TestAcc@5')
            with torch.no_grad():
                for images, target in test_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 5))
                    test_top1.update(acc1[0].item(), images.size(0))
                    test_top5.update(acc5[0].item(), images.size(0))

            #best_val_acc.top1 = max(best_val_acc.top1, val_top1.avg)
            best_val_acc.top5 = max(best_val_acc.top5, val_top5.avg)
            if val_top1.avg > best_val_acc.top1: 
                best_val_acc.top1 = val_top1.avg
                # keeping test_acc at the maximum val top1 acc
                best_test_acc.top1 = max(best_test_acc.top1, test_top1.avg)
                best_test_acc.top5 = max(best_test_acc.top5, test_top5.avg)
            
            # dump epoch stats
            stats = dict(epoch=epoch, 
                        val_acc1=val_top1.avg, val_acc5=val_top5.avg, best_val_acc1=best_val_acc.top1, best_val_acc5=best_test_acc.top5,
                        test_acc1=test_top1.avg, test_acc5=test_top5.avg, best_test_acc1=best_test_acc.top1, best_test_acc5=best_test_acc.top5)
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

        # sanity check
        if args.weights == 'freeze':
            reference_state_dict = torch.load(args.pretrained, map_location='cpu')
            model_state_dict = model.module.state_dict()
            for k in reference_state_dict:
                assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k]), k

        scheduler.step()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1, best_val_acc=best_val_acc, best_test_acc=best_test_acc,
                model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


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
