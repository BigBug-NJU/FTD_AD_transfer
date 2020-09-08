'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from mw_setting import parse_opts 
from datasets.nifd import NiiFolder
#from datasets.brains18 import BrainS18Dataset 
#from model import generate_model
#from trans_model import generate_model
from mw_model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os
import shutil

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

def train_one_epoch(data_loader, model, criterion, optimizer, total_epochs, save_interval, save_folder, sets, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    model.train()

    for batch_id, (volumes, target) in enumerate(data_loader):
        data_time.update(time.time() - end)

        if not sets.no_cuda: 
            volumes = volumes.cuda()
            target = target.cuda()
        
        output = model(volumes)
        #print("train:", output.shape, "-->", target.shape)
        if output.ndim != 2 or target.ndim != 1:
            print("train:", output.shape, "-->", target.shape)
            continue
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), volumes.size(0))
        top1.update(acc1[0], volumes.size(0))
        top5.update(acc2[0], volumes.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_id % sets.print_freq == 0:
            progress.display(batch_id)

def validate(val_loader, model, criterion, sets):
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
        for batch_id, (volumes, target) in enumerate(val_loader):
            if not sets.no_cuda: 
                volumes = volumes.cuda()
                target = target.cuda()

            # compute output
            output = model(volumes)
            if output.ndim != 2 or target.ndim != 1:
                print("varfy:", output.shape, "-->", target.shape)
                continue
            
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), volumes.size(0))
            top1.update(acc1[0], volumes.size(0))
            top5.update(acc2[0], volumes.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_id % sets.print_freq == 0:
                progress.display(batch_id)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

best_acc1 = 0
def train(data_loader, val_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    global best_acc1
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    criterion = nn.CrossEntropyLoss()

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        criterion = criterion.cuda()
        
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        #scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
        train_one_epoch(data_loader, model, criterion, optimizer, total_epochs, save_interval, save_folder, sets, epoch)
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, sets)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        #
        scheduler.step()                    
    print('Finished training')            
    if sets.ci_test:
        exit()

def save_checkpoint(state, is_best, filename='mw_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'mw_model_best.pth.tar')

if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    if sets.ci_test:
        sets.img_list = './toy_data/test_ci.txt' 
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 14
        sets.input_H = 28
        sets.input_W = 28

    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 

    # print(model)

    # optimizer
    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
                ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True 
    traindir = os.path.join(sets.data_root, 'train')
    valdir = os.path.join(sets.data_root, 'val')
    training_dataset = NiiFolder(traindir, sets)
    val_dataset = NiiFolder(valdir, sets)
    #training_dataset = BrainS18Dataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    # training
    train(data_loader, val_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 

# e3d_mw
# python mw_transfer.py --gpu_id 2 3 --data_root=../../data/mwc1/ --num_classes=3 --pretrain_path=./pretrain/resnet_50_23dataset.pth --batch_size=4 --n_epochs=200 > hjj_mw.log