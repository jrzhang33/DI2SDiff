

import random
import numpy as np
import torch
import sys
import os
import argparse
import torchvision
import PIL


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append('eval%d_in' % i)
            eval_name_dict['valid'].append('eval%d_out' % i)
        else:
            eval_name_dict['target'].append('eval%d_out' % i)
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'diversify': ['class', 'dis', 'total']}
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def act_param_init(args):

    tmp = {'pamap':((27, 1, 512),8, 10),'uschad':((6, 1, 500),12, 10),'dsads':((45, 1, 125),19, 10)}
    args.num_classes, args.input_shape, args.grid_size = tmp[
        args.dataset][1], tmp[args.dataset][0], tmp[args.dataset][2]

    return args


def get_args(dataset,args):

        # parser.add_argument('--algorithm', type=str, default="di2")
        # parser.add_argument('--beta1', type=float, default=0.5, help="Adam")
        # parser.add_argument('--projection', type=int, default=64) 
        # parser.add_argument('--classifier', type=str,
        #                     default="linear", choices=["linear", "wn"])
        # parser.add_argument('--data_file', type=str, default='')
        # parser.add_argument('--dis_hidden', type=int, default=256)
        # parser.add_argument('--weight_decay', type=float, default=5e-4)
        # parser.add_argument('--layer', type=str, default="bn",
        #                     choices=["ori", "bn"])

    args.algorithm = "di2"
    args.beta1 = 0.5
    args.projection = 64
    args.classifier = "linear"
    args.data_file = ""
    args.dis_hidden = 256
    args.weight_decay = 5e-4
    args.layer = "bn"


    if args.dataset == 'uschad':
        args.projection = 128
    if args.dataset == 'dsads' and args.target == 3:
        args.weight_decay = 2e-2
    if args.dataset == 'dsads' and args.target == 0:
        args.weight_decay = 5e-3
    if args.dataset == 'uschad' and args.target == 4:
        args.weight_decay = 1e-5

    args = act_param_init(args)
    args.drop = 1000
    args.drop_rate = 0.5

    return args
