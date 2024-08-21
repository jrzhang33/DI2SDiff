import torch
import sys
sys.path.append('./DI2SDiff/')
sys.path.append('./Diffusion_model/')
sys.path.append('./Style_conditioner/')
import os
import numpy as np
from datetime import datetime
import argparse
from Style_conditioner.trainer.trainer import model_load
from models.TC import TC
from models.model import base_Model
# Args selections
start_time = datetime.now()

import sys
sys.path.append('./')




import importlib
def one_hot(y, num_classes):
  y = y.long()
  return torch.eye(num_classes).cuda()[y]
def conditioner(x, y, testuser, dataset='uschad'):
    # args = parser.parse_args()
    device = 'cuda'
    selected_dataset = testuser['name'].split('_tar')[0]

    data_type = selected_dataset

    # Dynamically import the module
    module_name = f'config_files.{data_type}_Configs'
    ConfigModule = importlib.import_module(module_name)
    
    # Access the Config class from the imported module
    configs = ConfigModule.Config()
    configs.TC.train_test = 1 #test

    if x.shape[2] == 1:
        x = x.squeeze(2)
    configs.batch_size = x.shape[0]
    model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)
    c_t = model_load(model, temporal_contr_model, x, device, configs, testuser)
    return c_t
