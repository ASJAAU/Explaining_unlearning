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
        kernel, bias = conv.weight.data, conv.bias.data

        print(f"Layer {i}: {kernel.shape}")


        if i == 3:
            break