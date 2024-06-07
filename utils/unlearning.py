import torch
from torch import nn

import numpy as np
import sys
from tqdm import tqdm


def confuse_vision(model, noise_scale, trans = True, reinit_last = True, train_dense = True):
    """ Add Gaussian noise to the conv2d layers of the model.
        - model: a tf model with loaded weights
        - noise_scale: scale of the std of the Gaussian noise
        - trans: transpose kernel of cnn?
        - reinit_last: reinitialize last layer? otherwise add noise
    """
    #Get conv2d layers
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    print(f"Adding noise to {len(conv_layers)} convolutional layers")

    #Add noise to each kernel
    for i, conv in tqdm(enumerate(conv_layers), total=len(conv_layers)):
        print(conv)
        kernel = torch.clone(conv.weight)
        
        if conv.bias:
            bias = torch.clone(conv.bias)
            print(f"Layer {i}: {bias.shape}")
        else: 
            bias = nn.Parameter(torch.zeros(conv.out_channels))
            #bias = "Anders"
            conv.bias = bias


        print(f"Layer {i}: {kernel.shape}")
        conv.weight = nn.Parameter(torch.zeros(kernel.size()))
        print(conv.weight)
        if i == 1:
            break

    
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    for i, conv in tqdm(enumerate(conv_layers), total=len(conv_layers)):
        print(conv.weight)
        print(conv.bias)
        if i == 1: 
            break

def forget_loss():
    pass