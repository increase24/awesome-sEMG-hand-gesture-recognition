import os
os.sys.path.append('.')
import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader
from datasets.ninapro import Ninapro

def get_dataloader_db1(cfg, path_s):
    seq_lens = cfg.seq_lens
    step = cfg.step
    emgs = np.loadtxt(os.path.join(path_s, 'emg.txt'))
    labels = np.loadtxt(os.path.join(path_s, 'restimulus.txt'))
    repetitions = np.loadtxt(os.path.join(path_s, 'rerepetition.txt'))

    # perform 1-order 1Hz low-pass filter
    order = 1
    fs = 100 # sample rate: 100Hz
    cutoff = 1 # cutoff frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, 'lowpass')
    for _col in range(emgs.shape[1]):
        emgs[:,_col] = signal.filtfilt(b, a, emgs[:,_col])
    
    # u-law normalization
    u = 256
    emgs = np.sign(emgs) * np.log(1+u*abs(emgs))/np.log(1+u)
    
    # segmentation of training and testing samples
    length_dots = len(labels)
    data_train = []
    labels_train = []
    data_val = []
    labels_val = []
    for seq_len in seq_lens:
        for idx in range(0, length_dots - length_dots%seq_len, step):
            if labels[idx]>0 and labels[idx+seq_len-1]>0 and labels[idx]==labels[idx+seq_len-1]:
                repetition = repetitions[idx]
                if repetition in [2,5,7]: # val dataset
                    data_val.append(emgs[idx:idx+seq_len,:])
                    labels_val.append(labels[idx])
                else: # train dataset
                    data_train.append(emgs[idx:idx+seq_len,:])
                    labels_train.append(labels[idx])
    trainset = Ninapro(data_train, labels_train)
    valset = Ninapro(data_val, labels_val)
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(valset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    with open('./cfgs/db1.yaml') as cfg_file:
        cfg = CN.load_cfg(cfg_file)
        print('Successfully loading the config file...')
        dataCfg = cfg['DatasetConfig']
        paths_s = glob.glob(os.path.join(dataCfg.root_path, 's1'))
        train_loader, val_loader = get_dataloader_db1(dataCfg, paths_s[0])
        print('Successfully get dataloader of Ninapro dataset...')
        emg, label = iter(train_loader).next()
        print(emg.shape, label.shape)
