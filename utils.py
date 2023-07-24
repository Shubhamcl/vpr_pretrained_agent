from os import makedirs
from os.path import join
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def select_device(chosen_metal='cuda'):
    if 'cuda' in chosen_metal:
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loggers(log_dir, exp_name=""):
    # Add hydra saving here

    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name += exp_name
    log_dir = join(log_dir, folder_name)
    model_dir = join(log_dir, 'models')
    
    makedirs(log_dir)
    makedirs(model_dir)

    writer = SummaryWriter(log_dir)

    return model_dir, writer