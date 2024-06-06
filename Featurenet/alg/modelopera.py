
import torch
from network import act_network
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_fea_decoder(args):
    net_decoder = act_network.ActNetworkDecoder(args.dataset)
    return net_decoder

def get_fea(args):
    if args.dataset == 'uschad':
        net = act_network.ActNetwork_usc(args.dataset)
    else:
        net = act_network.ActNetwork(args.dataset)
    return net



