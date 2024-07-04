# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np


class DataDataset(object):
    def __init__(self, x,label, alabel, indices=None, dataset='dsads'):

        self.x = x
        self.label = label
        self.auglabel = alabel
        self.dataset = dataset
        if indices is None:
            self.indices = np.arange(len(self.x))


        self.transform = None
        self.target_transform = None

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x, self.dataset)
        else:
            return x

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def __getitem__(self, index):
        index = self.indices[index]
        xx = self.input_trans(self.x[index])
        clabel = self.target_trans(self.label[index])
        alabel = self.target_trans(self.auglabel[index])
    
        return xx, clabel,alabel

    def __len__(self):
        return len(self.indices)




class SensorDataset(object):
    def __init__(self, data, aug, indices=None, dataset='dsads'):
        self.data = data
        self.x = self.data[0]
        self.label = self.data[1]
        
        self.dataset = dataset
        if indices is None:
            self.indices = np.arange(len(self.x))
        if aug == True:
            self.aug = True
            self.auglabel = self.data[2]
        else:
            self.aug = False
            if len(self.data) == 3:
                self.slabel = self.data[2]
            self.auglabel = np.ones(self.label.shape) * 7
        self.transform = None
        self.target_transform = None

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x, self.dataset)
        else:
            return x

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def __getitem__(self, index):
        index = self.indices[index]
        xx = self.input_trans(self.x[index])
        clabel = self.target_trans(self.label[index])
        alabel = self.target_trans(self.auglabel[index])
        if len(self.data) == 3 and self.aug == False:
            slabel = self.target_trans(self.slabel[index])
            return xx, clabel, alabel,slabel
        else:
            return xx, clabel, alabel

    def __len__(self):
        return len(self.indices)
