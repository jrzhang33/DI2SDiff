import torch
from denoising_diffusion_pytorch import Trainer1D_Train as Trainer1D, Unet1D_cond_train as Unet1D_cond, GaussianDiffusion1Dcond_train as GaussianDiffusion1Dcond
import os
import sys
sys.path.append('./')
from data_load.get_domainhar import get_acthar
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()

parser.add_argument('--seed', default=1, type=int,
                    help='seed value')
parser.add_argument('--selected_dataset', default='uschad', type=str,
                    help='Dataset of choice: pamap, uschad, dsads')
parser.add_argument('--target', default=4,type=int,
                    help='Choose task id')
parser.add_argument('--remain_rate', default=0.2,
                    help='Using training data ranging from 0.2 to 1.0')
parser.add_argument('--results_folder', default='./Diffusion_model/dm_pth/', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--batch_size', default=64,type=int,
                    help='Training batch')
args = parser.parse_args()




#prepare some key info 
data_type = args.selected_dataset
target = args.target
remain_rate =args.remain_rate
testuser = {}
testuser['seed'] = args.seed
testuser['name'] = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])

conditioner = os.getcwd()+os.path.join('/Style_conditioner/conditioner_pth/')
testuser['conditioner'] =  conditioner + data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])+'-'+f'ckp_last-dl.pt'
train_loader, valid_loader, target_loader,testuser['n_class']  = get_acthar(args, data_type ,target , batch_size =64, remain_rate = remain_rate, seed = testuser['seed'])
source_loaders= train_loader


# Remainig data
testuser['remain_data'] = remain_rate
print("Remain:",testuser['remain_data'])

for minibatch in source_loaders:
    batch_size = minibatch[0].shape[0]
    print("print shape X:",minibatch[0].shape)
    shapex =[minibatch[0].shape[0],minibatch[0].shape[1],minibatch[0].shape[2]]#length,channel
    break
if shapex[1] % 64 !=0: # Pad the length for Unet
    shapex[1] =  64 - (shapex[1] % 64) +shapex[1]

model_our = Unet1D_cond(    
    dim = 64,
    num_classes = 100,  #style condition embedding dim
    dim_mults = (1, 2, 4, 8),
    channels = shapex[2],
    context_using = True #use style condition
)

diffusion = GaussianDiffusion1Dcond(
    model_our ,
    seq_length = shapex[1],
    timesteps = 100,  
    objective = 'pred_noise'
)
diffusion= diffusion.to(device)
train_loader = source_loaders
trainer = Trainer1D(
    diffusion,
    dataloader = train_loader,
    train_batch_size = shapex[0],
    train_lr = 2e-4,
    train_num_steps = 60001  ,       # total training steps1，000，000
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision 32->16
    results_folder=args.results_folder 
)

trainer.train(testuser)


