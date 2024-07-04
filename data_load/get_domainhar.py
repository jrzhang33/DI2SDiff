# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.utils.data as data
from data_load.data_util.get_dataloader import load

import data_load.utils
import os
# def args_parse():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_workers', type=int, default=1)
#     parser.add_argument('--seed', type=int, default=1)
#     parser.add_argument('--dataset', type=str, default='dsads')
#     parser.add_argument('--target', type=int, default=0)
#     parser.add_argument('--n_act_class', type=int, default=19,
#                         help='the number of the category of activities')
#     parser.add_argument('--n_aug_class', type=int,
#                         default=8, help='including the ori one')
#     parser.add_argument('--auglossweight', type=float, default=1)
#     parser.add_argument('--conweight', type=float, default=1.0)
#     parser.add_argument('--dp', type=str, default='dis',
#                         help='this is for oirginal and aug feature discrimination')
#     parser.add_argument('--dpweight', type=float, default=10.0,
#                         help='this is the weight of dp')
#     parser.add_argument('--n_feature', type=int, default=64)
#     parser.add_argument('--lr', type=float, default=0.0008)
#     parser.add_argument('--n_epoch', type=int, default=500)
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--early_stop', type=int, default=20)
#     parser.add_argument('--epoch_based_training', type=str2bool, default=False,
#                         help="Epoch-based training / Iteration-based training")
#     parser.add_argument('--n_iter_per_epoch', type=int,
#                         default=200, help="Used in Iteration-based training")
#     parser.add_argument('--remain_data_rate', type=float, default=1.0,
#                         help='The percentage of data used for training after reducing training data.')
#     parser.add_argument('--scaler_method', type=str, default='norm') #minmax
#     parser.add_argument('--root_path', type=str,
#                         default="/home/ddlearn/data/")
#     parser.add_argument('--data_save_path', type=str,
#                         default="/home/ddlearn/data/")
#     parser.add_argument('--save_path', type=str,
#                         default="/home/results/")

#     args = parser.parse_args()
#     args.step_per_epoch = 100000000000
#     args = param_init(args)
#     print_environ()
#     return args



def get_acthar(args, dataset = 'pamap',target = 0, batch_size =64, remain_rate = 1, seed = 1, train_diff = 0):
    # args = args_parse()
    args.step_per_epoch = 100000000000
    args.scaler_method = 'norm'
    args.batch_size = batch_size
    args.num_workers = 1
    args.root_path = os.getcwd() + '/data/'
    args.seed = seed
    if dataset == 'pamap':
        args.dataset = dataset #'pamap',uschad
        args.target = target
        args.n_act_class = 8
        args.auglossweight = 10.0
        args.conweight = 10.0
        args.dp = 'dis'
        args.dpweight = 0.1
        args.n_feature = 64
        args.remain_data_rate = remain_rate
    elif dataset == 'uschad':
        args.dataset = dataset 
        args.target = target
        args.n_act_class = 12
        args.auglossweight = 10.0
        args.conweight = 10.0
        args.dp = 'dis'
        args.dpweight = 0.1
        args.n_feature = 128
        args.remain_data_rate = 1.0
        args.remain_data_rate = remain_rate
    elif dataset == 'dsads':
        args.dataset = dataset 
        args.target = target
        args.n_act_class = 19
        args.auglossweight = 10.0
        args.conweight = 10.0
        args.dp = 'dis'
        args.dpweight = 0.1
        args.n_feature = 128
        args.remain_data_rate = remain_rate
    print(args)
    data_load.utils.set_random_seed(args.seed)
    train_ori_loader, train_aug_loader, val_ori_loader, val_aug_loader, test_ori_loader, test_aug_loader = load(
        args)
    if train_diff:
        train_dataset = train_ori_loader.dataset
        valid_dataset = val_ori_loader.dataset
        test_dataset = test_ori_loader.dataset
        combined_dataset = data.ConcatDataset([train_dataset, valid_dataset, test_dataset ])  
        source_loaders= data.DataLoader(combined_dataset, batch_size=16, drop_last=True, shuffle=True) 
        
        return source_loaders,val_ori_loader,test_ori_loader,args.n_act_class
    else:
        return train_ori_loader,val_ori_loader,test_ori_loader,args.n_act_class
   
