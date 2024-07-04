import torch

import sys
sys.path.append('.')
sys.path.append('./Diffusion_model/')
from denoising_diffusion_pytorch import Unet1D_cond,GaussianDiffusion1Dcond
from data_load.get_domainhar import get_acthar
from data_load.get_domainhar import get_acthar
import torch.nn as nn
import torch.utils.data as data
from train_strategy import train_diversity
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Featurenet.utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
import os
import argparse
import numpy as np
from Featurenet.config_files.distrb_condition import cond_set

def set_random_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()

parser.add_argument('--seed', default=1, type=int,
                    help='seed value')
parser.add_argument('--dataset', default='pamap', type=str,
                    help='Dataset of choice: pamap, uschad, dsads')
parser.add_argument('--target', default=1,type=int,
                    help='Choose task id')
parser.add_argument('--remain_rate', default=0.2,type=float,
                    help='Using training data ranging from 0.2 to 1.0')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--batch_size', default=64,type=int,
                    help='Training batch')
parser.add_argument('--Ocomb', default=5,type=int,
                    help='the maxmium of combination')

parser.add_argument('--Ktimes', default=2,type=int,
                    help='the ratio between new and ori')
parser.add_argument('--lr_decay_cls_f', default=1e-1,type=float,
                    help='step-3-fea')
parser.add_argument('--lr_decay_cls', default=1e-1,type=float,
                    help='step-3-net')
parser.add_argument('--lr_decay_ori', default=1e-1,type=float,
                    help='step-2-net')
parser.add_argument('--lr_decay_ori_f', default=1e-1,type=float, #1e-1
                    help='step-2-fea')
parser.add_argument('--lr_decay1', default=1e-2,type=float, #1e-2
                    help='step-1-fea')
parser.add_argument('--lr_decay2', default=1,type=float,
                    help='step-1-net')
parser.add_argument('--lr', type=float, default=7e-3, help="learning rate")
parser.add_argument('--step1', default=0,type=int,
                    help='epoch1')
parser.add_argument('--step2', default=0,type=int,
                    help='epoch2')
parser.add_argument('--step3', default=1,type=int,
                    help='epoch3')

args = parser.parse_args()
   


#Some key info
target = args.target
data_type  = args.dataset
remain_rate =args.remain_rate
testuser = {}
batch_size = args.batch_size
testuser['target'] = target
testuser['maxcond'] =args.Ocomb
testuser['cond_weight'] = cond_set(data_type, remain_rate, target, testuser['maxcond'])
testuser['dataset'] = data_type
testuser['repeat'] = args.Ktimes #repeat diverse samples for k times 
testuser['seed']  = args.seed
set_random_seed(testuser['seed'])
testuser['remain_data'] = args.remain_rate
print("Remain:",testuser['remain_data'])
testuser['name'] = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])
dataset =testuser['name'].split('_tar')[0] 
args_data = get_args(dataset,args)


# for attr, value in vars(args_data).items():
#     setattr(args, attr, value)

conditioner = os.getcwd()+os.path.join('/Style_conditioner/conditioner_pth/')

# Load data
train_loader, valid_loader, target_loader,testuser['n_class'] = get_acthar(args,data_type ,target , batch_size =64,remain_rate = testuser['remain_data'], seed = testuser['seed'])
train_dataset = train_loader.dataset
valid_dataset = valid_loader.dataset
source_loaders= data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
valid_loader= data.DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
testuser['conditioner'] =  conditioner + data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])+'-'+f'ckp_last-dl.pt'

# Load style conditioner, diffusion and newdata_path

testuser['newdata']  = os.getcwd()+'/Featurenet/new_data/' +testuser['name']+'-rep'+str(testuser['repeat'])+'-batch'+str(batch_size)+'-cond'+str(testuser['maxcond']) +'-weight'+(str(testuser['cond_weight']))+'.pt'
testuser['diff'] = os.getcwd()+'/Diffusion_model/dm_pth/'+testuser['name']+'.pt'

for minibatch in source_loaders:
    batch_size = minibatch[0].shape[0]
    print("print shape X:",minibatch[0].shape)
    shape=[minibatch[0].shape[1],minibatch[0].shape[2]]#length,channel
    break

testuser['length'] = shape[0]
if shape[0] % 64 !=0:
    shape[0] =  64 - (shape[0] % 64) +shape[0]


model_our = Unet1D_cond(    
    dim = 64,
    num_classes = 100, 
    dim_mults = (1, 2, 4, 8),
    channels = shape[1],
    context_using = True
)

diffusion = GaussianDiffusion1Dcond(
    model_our ,
    seq_length = shape[0],
    timesteps = 100,  
    objective = 'pred_noise'
)

diffusion= diffusion.to(device)
diffusion= diffusion.to(device)
data = torch.load(testuser['diff'])
diffusion.load_state_dict(data['model'])
criterion = nn.CrossEntropyLoss()


train_diversity(diffusion,args,source_loaders, valid_loader, target_loader,testuser) #fine_tune using generative samples
print(testuser['name'])
print(testuser['newdata'])

