import pickle
import torch
import shutil
import os
import numpy as np

def save_result(result_npy, output_dir, save_npy_name):
    print("Saving validation results ...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, save_npy_name)
    np.savetxt(model_path, result_npy)
        

def save_checkpoint(state, output_dir, saveName_bstModel='model_best.pth.tar'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, saveName_bstModel)
    torch.save(state, model_path)