# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from .algs.convnetwork import FeatureNet

ALGORITHMS = [
    'featurenetwork'
]


def get_algorithm_class():
    return FeatureNet
