import torch
from torch import nn

import numpy as np
import sys
from tqdm import tqdm


def confuse_vision(model, noise_scale = 0.1, trans = True, reinit_last = True, train_dense = True, train_kernel = True, train_bias = True, train_last = True):
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
        print(f"Convolution layer {i}: {conv}")

        #Confusing kernels
        kernel = torch.clone(conv.weight)
        for j in tqdm(range(kernel.shape[0]), total=kernel.shape[0]):
            for k in range(kernel.shape[1]):
                # print(f"Layer {i}: Kernel ({j},{k})")
                # print(kernel[j,k,:,:])
                # print(kernel[j,k,:,:].shape)
                #Traspose kernel
                if trans:
                    kernel[j,k,:,:] = torch.transpose(torch.clone(kernel[j,k,:,:]), 0, 1)
                # print(kernel[j,k,:,:])
                # print(kernel[j,k,:,:].shape)
                #Compute noise scale proportional to the std of the kernel
                noise_std = torch.std(kernel[j,k,:,:]).item() * noise_scale
                # print(noise_std)
                if not noise_std > 0:
                    # print("No noise added")
                    noise_std = 0.01
                #Add noise
                kernel[j,k,:,:] += torch.normal(mean=0, std=noise_std, size=kernel[j,k,:,:].shape, device=kernel.device)
                # print(kernel[j,k,:,:])
                # print(kernel[j,k,:,:].shape)
        #Update kernel
        conv.weight = nn.Parameter(kernel, requires_grad=train_kernel)

        #Confusing bias
        if conv.bias:
            # print("There is bias")
            bias = torch.clone(conv.bias)
            # print(f"Layer {i}: {bias.shape}")

            #Compute noise scale proportional to the std of the bias
            noise_std = torch.std(bias).item() * noise_scale
            if not noise_std > 0:
                noise_std = 0.01
            #Add noise
            bias += torch.normal(mean=0, std=noise_std, size=bias.shape, device=bias.device)
            #Update bias
            conv.bias = nn.Parameter(bias, requires_grad=train_bias)
        # else: 
            # print("No bias")
            # bias = nn.Parameter(torch.zeros(conv.out_channels))
            # conv.bias = bias

        #Break for debugging
        # if i == 1:
        #     break

    module_list = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
    
    #Set non conv2d layers to trainable/not trainable
    for i, m in enumerate(module_list):
        if not isinstance(m, nn.Conv2d):
            m.requires_grad = train_dense
        if i == len(module_list) - 1:
            #Reinitialize last layer
            if reinit_last:
                if isinstance(m, nn.Linear):
                    print(f"Reinitializing last layer: {m}")
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                    m.requires_grad = train_last
                else: 
                    print(f"Last layer is not linear: {m}")
                    raise ValueError("Last layer is not linear")
            #Otherwise add Gaussian noise
            else:
                if isinstance(m, nn.Linear):
                    print(f"Adding noise to last layer: {m}")
                    weight = torch.clone(m.weight)
                    #Compute noise scale proportional to the std of the weight
                    noise_std = torch.std(weight).item() * noise_scale
                    if not noise_std > 0:
                        noise_std = 0.01
                    #Add weight noise
                    weight += torch.normal(mean=0, std=noise_std, size=weight.shape, device=weight.device)
                    #Update weight
                    m.weight = nn.Parameter(weight, requires_grad=train_last)

                    if m.bias:
                        print("There is bias in last layer")
                        bias = torch.clone(m.bias)
                        #Compute noise scale proportional to the std of the bias
                        noise_std = torch.std(bias).item() * noise_scale
                        if not noise_std > 0:
                            noise_std = 0.01
                        #Add bias noise
                        bias += torch.normal(mean=0, std=noise_std, size=bias.shape, device=bias.device)
                        #Update weight and bias
                        m.bias = nn.Parameter(bias, requires_grad=train_last)
                else: 
                    print(f"Last layer is not linear: {m}")
                    raise ValueError("Last layer is not linear")
            

    #Check if noise was added correctly debug
    # conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    # for i, conv in tqdm(enumerate(conv_layers), total=len(conv_layers)):
    #     print(conv.weight)
    #     print(conv.bias)
    #     if i == 1: 
    #         break

def forget_loss():
    pass