import torch
from .stats import AverageMeter, accuracy
import numpy as np

class Trainer(object):
    def __init__(self, train_loader, valid_loader, model, device, criterion, optimizer, print_freq):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.bst_acc = 0.0
        self.flag_improve = False

    def reset_optimiser(self, optimizer):
        self.optimizer = optimizer

    def train_epoch(self, epoch):
        losses = AverageMeter()
        accuracies = AverageMeter()
        self.model.train()
        for idx, (input, target) in enumerate(self.train_loader):
            # compute output and loss
            input, target = input.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(input)
            loss = self.criterion(output, target)
            losses.update(loss.item(), input.size(0))
            # measure accuracy
            [acc] = accuracy(output.detach(), target.detach().cpu())
            accuracies.update(acc.item(), input.size(0))
            # compute grandient and do back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        epoch, idx, len(self.train_loader),
                        loss = losses, acc = accuracies 
                    ))
        print('Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                    epoch, idx, len(self.train_loader),
                    loss = losses, acc = accuracies 
                ))
        return losses.avg, accuracies.avg
                
                
    def validate(self, eval_only = False):
        losses = AverageMeter()
        accuracies = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for idx, (input, target) in enumerate(self.valid_loader):
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input)
                loss = self.criterion(output, target)
                [acc] = accuracy(output.detach(), target.detach().cpu())
                losses.update(loss.item(), input.size(0))
                accuracies.update(acc.item(), input.size(0))
                if idx % self.print_freq == 0:
                    print(
                        'Test: [{0}/{1}]\t'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        idx, len(self.valid_loader),
                        loss = losses, acc = accuracies 
                    ))
            print('Test: [{0}/{1}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                idx, len(self.valid_loader),
                loss = losses, acc = accuracies 
            ))
            if(accuracies.avg>self.bst_acc):
                self.bst_acc = accuracies.avg
                self.flag_improve = True
            else:
                self.flag_improve = False
        return losses.avg, accuracies.avg