import os
os.sys.path.append('.')
import argparse
import glob
import shutil
import time
import json
import numpy as np
import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
# custom
from datasets.db1 import get_dataloader_db1
from networks import get_network
from networks.BiTCN import UiTCN, BiTCN 
from utils.trainer import Trainer
from utils.saver import save_checkpoint, save_result
from utils.initializer import weight_init


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training of sEMG based gesture recognition')
    parser.add_argument('--config', default='./cfgs/db1.yaml', help='json config file path')
    parser.add_argument('--model', default='BiTCN', help='the name of empolyed model')
    args = parser.parse_args()
    return args

def main():
    train_start_time = time.time()
    curr_time = time.strftime('%m-%d-%H-%M', time.localtime())
    args = parse_args()
    with open(args.config) as cfg_file:
        cfg = CN.load_cfg(cfg_file)
        opts = [
            'ModelConfig.model_name', args.model
            ]
        cfg.merge_from_list(opts)
        print('Successfully loading the config file....')
    ModelConfig = cfg.ModelConfig
    DataConfig = cfg.DatasetConfig
    OutputConfig = cfg.OutputConfig
    device = torch.device('cuda:0')
    paths_subjects = sorted(glob.glob(os.path.join(DataConfig.root_path,'*')))
    results_p = np.zeros((len(paths_subjects), 1))
    results_r = np.zeros((len(paths_subjects), 1))
    results_bi = np.zeros((len(paths_subjects), 1))
    # if os.path.exists(os.path.join(OutputConfig.dir_results, ModelConfig.model_name, ModelConfig.model_name+ f'_{ModelConfig.modality}.txt')):
    #     results = np.loadtxt(os.path.join(OutputConfig.dir_results, ModelConfig.model_name, ModelConfig.model_name+ f'_{ModelConfig.modality}.txt'))
    # start_idx_subject, end_idx_subject = (0, 8)
    for idx_subject, path_subject in enumerate(paths_subjects):
        print(f">>> index: {idx_subject}, subject_files: {path_subject}")
        # load dataloader
        print("Start loading the dataloader....")
        train_loader, val_loader = get_dataloader_db1(DataConfig, path_subject)
        print('Finish loading the dataloader....')
        # load network
        with open(ModelConfig.model_arch[ModelConfig.model_name]) as data_file:
            cfg_model = CN.load_cfg(data_file)
        
        # stage 1 (positive)
        model = UiTCN(cfg_model, reverse=False)
        model.apply(weight_init)
        model.to(device)
        # define criterion, optimizer, scheduler
        OptimizerConfig = cfg.OptimizerConfig
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=OptimizerConfig.lr)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60], gamma=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.5,patience=5,eps=1e-08)
        # start training
        num_epoches = int(OptimizerConfig.epoches)
        trainer = Trainer(train_loader, val_loader, model, device, criterion, optimizer, print_freq=100)           
        print(" > Training (stage 1, positive) is getting started...")
        print(" > Training (stage 1, positive) takes {} epochs.".format(num_epoches))
        # trainer.reset_optimiser(optimizer)
        for epoch in range(num_epoches):
            # train one epoch
            epoch_start_time = time.time()
            train_loss, train_acc = trainer.train_epoch(epoch) 
            valid_loss, valid_acc = trainer.validate(eval_only=False)
            epoch_end_time = time.time()
            print("epoch cost time: %.4f min" %((epoch_end_time - epoch_start_time)/60))
            print(f'current best accuracy: {trainer.bst_acc}')
            # remember best acc and save checkpoint
            if(trainer.flag_improve):
                print(f'the best accuracy increases to {trainer.bst_acc}')
                save_checkpoint({
                    'epoch': epoch,
                    'arch': ModelConfig['model_name'],
                    'state_dict': trainer.model.feat_net.state_dict(),
                    'best_acc': trainer.bst_acc}, 
                    os.path.join(OutputConfig.path_weights, ModelConfig.model_name),
                    ModelConfig.model_name+f'_{DataConfig.dataset}_s{idx_subject}_p'+'.pth.tar')
            scheduler.step(valid_acc)
        results_p[idx_subject] = trainer.bst_acc
        save_result(results_p, os.path.join(OutputConfig.path_results, ModelConfig.model_name), \
            f'{ModelConfig.model_name}_{DataConfig.dataset}_bs{DataConfig.batch_size}_p.txt')

        # stage 1 (reverse)
        model = UiTCN(cfg_model, reverse=True)
        model.apply(weight_init)
        model.to(device)
        # define criterion, optimizer, scheduler
        OptimizerConfig = cfg.OptimizerConfig
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=OptimizerConfig.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.5,patience=5,eps=1e-08)
        # start training
        num_epoches = int(OptimizerConfig.epoches)
        trainer = Trainer(train_loader, val_loader, model, device, criterion, optimizer, print_freq=100)           
        print(" > Training (stage 1, reverse) is getting started...")
        print(" > Training (stage 1, reverse) takes {} epochs.".format(num_epoches))
        # trainer.reset_optimiser(optimizer)
        for epoch in range(num_epoches):
            # train one epoch
            epoch_start_time = time.time()
            train_loss, train_acc = trainer.train_epoch(epoch) 
            valid_loss, valid_acc = trainer.validate(eval_only=False)
            epoch_end_time = time.time()
            print("epoch cost time: %.4f min" %((epoch_end_time - epoch_start_time)/60))
            print(f'current best accuracy: {trainer.bst_acc}')
            # remember best acc and save checkpoint
            if(trainer.flag_improve):
                print(f'the best accuracy increases to {trainer.bst_acc}')
                save_checkpoint({
                    'epoch': epoch,
                    'arch': ModelConfig['model_name'],
                    'state_dict': trainer.model.feat_net.state_dict(),
                    'best_acc': trainer.bst_acc}, 
                    os.path.join(OutputConfig.path_weights, ModelConfig.model_name),
                    ModelConfig.model_name+f'_{DataConfig.dataset}_s{idx_subject}_r'+'.pth.tar')
            scheduler.step(valid_acc)
        results_r[idx_subject] = trainer.bst_acc
        save_result(results_r, os.path.join(OutputConfig.path_results, ModelConfig.model_name), \
            f'{ModelConfig.model_name}_{DataConfig.dataset}_bs{DataConfig.batch_size}_r.txt')

        # stage 2 (bidirectional)
        model = BiTCN(cfg_model)
        #model.apply(weight_init)
        model.to(device)
        # load checkpoint
        OutputConfig = cfg.OutputConfig
        checkpoint_p = torch.load(os.path.join(OutputConfig.path_weights, ModelConfig.model_name, 
                ModelConfig.model_name+f'_{DataConfig.dataset}_s{idx_subject}_p.pth.tar'))
        checkpoint_r = torch.load(os.path.join(OutputConfig.path_weights, ModelConfig.model_name, 
                ModelConfig.model_name+f'_{DataConfig.dataset}_s{idx_subject}_r.pth.tar'))
        model.tcn_positive.load_state_dict(checkpoint_p['state_dict'])
        model.tcn_reverse.load_state_dict(checkpoint_r['state_dict'])
        # define criterion, optimizer, scheduler
        OptimizerConfig = cfg.OptimizerConfig
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=OptimizerConfig.lr/10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.5,patience=5,eps=1e-08)
        # start training
        num_epoches = int(OptimizerConfig.epoches)
        trainer = Trainer(train_loader, val_loader, model, device, criterion, optimizer, print_freq=100)           
        print(" > Training (stage 1, reverse) is getting started...")
        print(" > Training (stage 1, reverse) takes {} epochs.".format(num_epoches))
        # trainer.reset_optimiser(optimizer)
        for epoch in range(num_epoches):
            # train one epoch
            epoch_start_time = time.time()
            train_loss, train_acc = trainer.train_epoch(epoch) 
            valid_loss, valid_acc = trainer.validate(eval_only=False)
            epoch_end_time = time.time()
            print("epoch cost time: %.4f min" %((epoch_end_time - epoch_start_time)/60))
            print(f'current best accuracy: {trainer.bst_acc}')
            # remember best acc and save checkpoint
            if(trainer.flag_improve):
                print(f'the best accuracy increases to {trainer.bst_acc}')
                save_checkpoint({
                    'epoch': epoch,
                    'arch': ModelConfig['model_name'],
                    'state_dict': trainer.model.state_dict(),
                    'best_acc': trainer.bst_acc}, 
                    os.path.join(OutputConfig.path_weights, ModelConfig.model_name),
                    ModelConfig.model_name+f'_{DataConfig.dataset}_s{idx_subject}_bi'+'.pth.tar')
            scheduler.step(valid_acc)
        results_bi[idx_subject] = trainer.bst_acc
        save_result(results_r, os.path.join(OutputConfig.path_results, ModelConfig.model_name), \
            f'{ModelConfig.model_name}_{DataConfig.dataset}_bs{DataConfig.batch_size}_bi.txt')

    print('acc-avg_p:\n', np.mean(results_p))
    print('acc-avg_r:\n', np.mean(results_r))
    print('acc-avg_bi:\n', np.mean(results_bi))
    train_end_time = time.time()
    print("total training time: %.2f minutes" %((train_end_time - train_start_time)/60))

if __name__ == "__main__":
    main()
    