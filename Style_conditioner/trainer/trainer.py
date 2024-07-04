import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss
from datetime import datetime

time = datetime.now().strftime("%Y%m%d_%H%M%S")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_fc(x):
    x = x.squeeze(2).unsqueeze(3)
    b, c, d , e= x.size()
    x_fft, x_fft_init=transfft(x)
    k1=torch.Tensor(np.arange(1,x_fft.shape[2]+1)).to(device).repeat(x_fft.shape[0],x_fft.shape[1],1)
    fc=k1.unsqueeze(3)*x_fft
    y_1=(torch.sum(fc,dim=2)/torch.sum(x_fft,dim=2)).view(b,c,1,1) #fc
    return y_1.squeeze(3).squeeze(2)

def get_low(x):
    x = x.squeeze(2).unsqueeze(3)
    b, c, d , e= x.size()
    x_fft, x_fft_init=transfft(x)
    k1=torch.Tensor(np.arange(1,x_fft.shape[2]+1)).to(device).repeat(x_fft.shape[0],x_fft.shape[1],1)
    fc=k1.unsqueeze(3)*x_fft
    y_1=(torch.sum(fc,dim=2)/torch.sum(x_fft,dim=2)).view(b,c,1,1) #fc
    fc1=y_1.to(torch.int)
    x_= x_fft_init
    for i in range(x_.shape[0]):
        for j in range(x_.shape[1]):
            s = fc1[i,j,0,0].item()
           # s = int(x_.shape[2]/2)
            x_[i,j,s:]=0
    x2=torch.fft.irfft(x_,n=d,dim=2)
    return x2

def get_high(x):
    x = x.unsqueeze(3)
    b, c, d , e= x.size()
    x_fft, x_fft_init=transfft(x)
    k1=torch.Tensor(np.arange(1,x_fft.shape[2]+1)).to(device).repeat(x_fft.shape[0],x_fft.shape[1],1)
    fc=k1.unsqueeze(3)*x_fft
    y_1=(torch.sum(fc,dim=2)/torch.sum(x_fft,dim=2)).view(b,c,1,1) #fc
    fc1=y_1.to(torch.int)
    x_= x_fft_init
    for i in range(x_.shape[0]):
        for j in range(x_.shape[1]):
            s = fc1[i,j,0,0].item()
            #s = int(x_.shape[2]/2)
            x_[i,j,:s]=0
    x2=torch.fft.irfft(x_,n=d,dim=2)
    return x2


def transfft(x_input):
    x_fft_init = torch.fft.rfft(x_input[..., 0],dim=2,norm="forward")
    x_fft = torch.stack((x_fft_init.real, x_fft_init.imag), -1)
    x_fft = x_fft.detach().cpu().numpy()

    for i in range(x_fft.shape[0]): 
        if i==0:
            ff = np.sqrt(x_fft[i,:,:,0]**2 + x_fft[i,:,:,1]**2) 
            ff=ff.reshape(1,x_fft.shape[1],x_fft.shape[2],1)
            continue 
        f = np.sqrt(x_fft[i,:,:,0]**2 + x_fft[i,:,:,1]**2).reshape(1,x_fft.shape[1],x_fft.shape[2],1)
        ff=np.concatenate([ff,f],0)
    x_fft=torch.from_numpy(ff[:,:,:-1,:]).to(device)
    return x_fft,x_fft_init



def Trainer_ft(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode,testuser = None):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    best_acc = 0
    max_epochs = 50
    for epoch in range(1, max_epochs):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)
        if valid_acc > best_acc:
            best_acc = valid_acc
            chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, testuser+'-'+f'ckp_last-dl.pt'))
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    print("Best:", best_acc)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, testuser+'-'+f'ckp_last-dl.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")

def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode,testuser):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch):
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                    )
        chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, testuser+'-'+f'ckp_last-dl.pt'))
        print(os.path.join(experiment_log_dir,  testuser, f'ckp_last-dl.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")

from dataloader.augmentations import DataTransform

def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()
    for batch_idx, minibatch in enumerate(train_loader): # len(minibatch)=3, the last element indicates the type of the original data
        # send to device
        x = minibatch[0].cuda().float() #batch,len,channel,1
        y = minibatch[1].cuda().long()
    
        if len(x.shape) == 3:
            x = x.unsqueeze(3)
        data = x.transpose(1, 2).squeeze(3)
        aug1,aug2 = DataTransform(data.cpu(),config)
        aug1,aug2 = torch.tensor(aug1).clone().detach(), torch.tensor(aug2).clone().detach()
        data, y = data.float().to(device), y.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        
        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)
            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2  

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                        config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
            
        else: # supervised training or fine tuining
            output = model(data)
            predictions, features = output
            loss = criterion(predictions, y)
            total_acc.append(y.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc
   
   
  
def model_load(model, temporal_contr_model, x, device, config,testuser):
    model.eval()
    temporal_contr_model.eval()
    path = testuser['conditioner']
    chkpoint = torch.load(path, map_location=device)
    model_dict= chkpoint["model_state_dict"]
    tc_dict = chkpoint['temporal_contr_model_state_dict']
    model.load_state_dict(model_dict)
    temporal_contr_model.load_state_dict(tc_dict)
    with torch.no_grad():
        x = x.float().to(device)       
        predictions1, features1 = model(x)
        features1 = F.normalize(features1, dim=1)
        c_t = temporal_contr_model.context(features1)
    return c_t


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels,_ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            data = data.transpose(1, 2)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
