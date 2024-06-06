# For fairness,
# the network architecture used in our method is the same as the common time-series DG settings used in the library below
# https://github.com/microsoft/robustlearn


import torch.nn as nn


var_size = {
            'dsads': {
                'in_size': 45,
                'ker_size': 9,
                'fc_size': 32*25
            },
            'pamap': {
                'in_size': 27,
                'ker_size': 9,
                'fc_size': 32*122
            },

                        'uschad': {
                'in_size': 6,
                'ker_size': 6,
                'fc_size': 64*58
            },
        }

class ActNetworkDecoder(nn.Module):
    def __init__(self, taskname):
        super(ActNetworkDecoder, self).__init__()
        self.taskname = taskname
        self.in_features = var_size[taskname]['fc_size']
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(
                1, var_size[taskname]['ker_size']), stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=var_size[taskname]['in_size'], kernel_size=(
                1, var_size[taskname]['ker_size']), stride=2),
            nn.BatchNorm2d(var_size[taskname]['in_size']),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )

    def forward(self, x):
        x = x.view(-1, 32, self.in_features//(32), 1)  # Reshape back to the output of conv2
        x = self.conv1(self.conv2(x))
        return x

class ActNetwork(nn.Module):
    def __init__(self, taskname):
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=var_size[taskname]['in_size'], out_channels=16, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5)  
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5)  
        )
        self.in_features = var_size[taskname]['fc_size']

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        try:
            x = x.view(-1, self.in_features)
        except:
            x=  x.contiguous().view(-1, self.in_features)
        return x
import torch.nn.init as init
class ActNetwork_usc(nn.Module):
    def __init__(self, taskname):
        super(ActNetwork_usc, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=var_size[taskname]['in_size'], out_channels=16, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5) 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5)  
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5) 
        )
        self.in_features = var_size[taskname]['fc_size']
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.conv3(x)
        try:
            x = x.view(-1, self.in_features)
        except:
            x=  x.contiguous().view(-1, self.in_features)
        return x