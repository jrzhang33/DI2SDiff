# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.utils.data as data
from data_load.data_util.get_dataloader import load

import data_load.utils
import os




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
   
