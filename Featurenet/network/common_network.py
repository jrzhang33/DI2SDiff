# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm

class projectionDecoder(nn.Module):
    def __init__(self, projection_dim=256, feature_dim=None, type="ori"):
        super(projectionDecoder, self).__init__()
        self.bn = nn.BatchNorm1d(feature_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.projection = nn.Linear(projection_dim, feature_dim)
        self.type = type

    def forward(self, x):
        if self.type == "bn":
            x = self.bn(x)
        x = self.projection(x)
        return x



class feat_projection(nn.Module):
    def __init__(self, feature_dim, projection_dim=256, type="ori"):
        super(feat_projection, self).__init__()
        self.bn = nn.BatchNorm1d(projection_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.projection = nn.Linear(feature_dim, projection_dim)
        self.type = type

    def forward(self, x):
        x = self.projection(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, projection_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(projection_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(projection_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
