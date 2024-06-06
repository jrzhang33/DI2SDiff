
import torch
def get_slr(data_type, target, optf, opto, optc):
    if data_type == 'pamap':
        if target == 0:
            schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=20) #step 3
            schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=30) #step 1
            return schedulera, schedulerd, scheduler, False
        elif target == 1 or target == 2:
            schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=5) #step 3
            schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=15) #step 1
            return schedulera, schedulerd, scheduler, False       
        elif target == 3:
            schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
            scheduler = torch.optim.lr_scheduler.StepLR(optc, step_size=100, gamma=0.5)
            schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=15) #step 1
            return schedulera, schedulerd, scheduler, True   
    if data_type == 'dsads':
            if target == 0 :
                schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=20) #step 3
                schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=30) #step 1  
                return schedulera, schedulerd, scheduler, False     
            if target == 1:
                schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=15) #step 2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=30) #step 3
                schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=15) #step 1
                return schedulera, schedulerd, scheduler, False  
            if target == 2:
                schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=20) #step 3
                schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=30) #step 1  
                return schedulera, schedulerd, scheduler, False   
            if target == 3:
                schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=80) #step 3
                schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=30) #step 1  
                # schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
                # scheduler = torch.optim.lr_scheduler.StepLR(optc, step_size=50, gamma=0.5)
                # schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=15) #step 1
                return schedulera, schedulerd, scheduler, False     
    if data_type == 'uschad':
            if target == 0 :
                schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=30) #step 2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=30) #step 3
                schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=30) #step 1  
                return schedulera, schedulerd, scheduler, False     
            if target == 1:
                schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=15) #step 2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=30) #step 3
                schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=15) #step 1
                return schedulera, schedulerd, scheduler,  False      
            if target == 2:
                schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=20) #step 3
                schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=30) #step 1  
                return schedulera, schedulerd, scheduler, False   
            if target == 3:
                schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=5) #step 3
                schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=5) #step 1 
                return schedulera, schedulerd, scheduler, False     
            if target == 4:
                schedulerd = torch.optim.lr_scheduler.ReduceLROnPlateau(opto, 'min', patience=5) #step 2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optc, 'min', patience=5) #step 3
                schedulera = torch.optim.lr_scheduler.ReduceLROnPlateau(optf , 'min', patience=5) #step 1  
                return schedulera, schedulerd, scheduler, False           
def get_params(alg, args, nettype):
    init_lr = args.lr
    if nettype == 'step-2': #opto
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay_ori_f * init_lr},
            {'params': alg.dprojection.parameters(), 'lr':args.lr_decay_ori * init_lr},
            {'params': alg.dclassifier.parameters(), 'lr':args.lr_decay_ori * init_lr},
        ]
        return params
    elif nettype == 'step-3': #opc
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay_cls_f * init_lr},
            {'params': alg.projection.parameters(), 'lr': args.lr_decay_cls * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay_cls * init_lr},
        ]
        return params
    elif nettype == 'step-1': #optf
        params = [
            
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.aprojection.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.aclassifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
        return params


def get_optimizer(alg, args, nettype):
    params = get_params(alg, args, nettype=nettype)
    optimizer = torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, 0.9))

    return optimizer
