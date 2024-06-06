# The code for the style conditioner mainly follows a strong time-series representation learning method, i.e.,  TS-TCC (https://github.com/emadeldeen24/TS-TCC/).
import torch
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from trainer.trainer import Trainer, Trainer_ft
from models.TC import TC
from models.model import base_Model

import sys
sys.path.append('./')
from data_load.get_domainhar import get_acthar
import torch.utils.data as data
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Ex    periment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=1, type=int,
                    help='seed value')
parser.add_argument('--selected_dataset', default='uschad', type=str,
                    help='Dataset of choice: pamap, uschad, dsads')
parser.add_argument('--remain_rate', default=0.2,
                    help='Using training data ranging from 0.2 to 1.0')
parser.add_argument('--target', default=4,type=int,
                    help='Choose task id')
parser.add_argument('--training_mode', default='self_supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear,rl')

parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--batch_size', default=64,type=int,
                    help='Training batch')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
training_mode = args.training_mode
run_description = args.run_description

os.makedirs(args.logs_save_dir, exist_ok=True)

#Some key info
data_type = args.selected_dataset
target = args.target
remain_rate =args.remain_rate
SEED = args.seed
testuser = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)
batch_size = args.batch_size
#Load data
train_loader, valid_loader, target_loader,_ = get_acthar(args, data_type,target , batch_size =args.batch_size, remain_rate = remain_rate ,seed = SEED, train_diff=0)
train_dataset = train_loader.dataset
valid_dataset = valid_loader.dataset
source_loaders= data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()
configs.batch_size = batch_size

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(args.logs_save_dir, data_type+str(remain_rate)+f"_seed_{SEED}")

os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = f"./data/{data_type}"



logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

if training_mode == "fine_tune":
    load_from = experiment_log_dir
    chkpoint = torch.load(os.path.join(load_from, testuser+"-ckp_last-dl.pt"), map_location=device)
    logs_save_dir = './Style_conditioner/conditioner_pth/'
    experiment_log_dir = os.path.join(logs_save_dir, data_type+str(remain_rate)+f"_seed_{SEED}")
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr , betas=(configs.beta1  , configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    logger.debug(f"Training time is : {datetime.now()-start_time}")
    Trainer_ft(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader , device, logger, configs, experiment_log_dir, training_mode,testuser)

else:
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader , device, logger, configs, experiment_log_dir, training_mode,testuser)
    logger.debug(f"Training time is : {datetime.now()-start_time}")
   