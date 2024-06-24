import torch
from torch import nn

import numpy as np
import sys
from tqdm import tqdm


def confuse_vision(model, noise_scale = 0.1, add_noise = True, trans = True, reinit_last = True, train_dense = True, train_kernel = True, train_bias = True, train_last = True):
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
                if add_noise:
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
            if add_noise:
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
            elif add_noise:
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

    return model



def sebastian_unlearn(model, sebastian_type="prune", quantile=0.99, train_weight = True, train_bias = True):
    """ 
    Forget a class by removing or reinitializing the 99% quantile of the weights and setting the bias to zero.
    Criteria of removing weights is based on the L1 norm of the weights.
    """
    
    # # Get all the layers of the model
    # module_list = [module for module in model.modules() if not isinstance(module, nn.Sequential)]

    # # Get a list of all the weights and set the bias to zero
    # for i, m in enumerate(module_list):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         # Get the weights
    #         weight = torch.clone(m.weight)
    #         # Get the bias
    #         bias = torch.clone(m.bias)
    #         # Get the L1 norm of the weights
    #         weight_l1 = torch.norm(weight, p=1, dim=1)
    #         # Get the 99% quantile of the L1 norm
    #         quantile_value = torch.quantile(weight_l1, quantile)
    #         # Get the indices of the weights that are less than the 99% quantile
    #         indices = weight_l1 < quantile_value
    #         # Set the weights to zero
    #         weight[indices] = torch.zeros_like(weight[indices])
    #         # Set the bias to zero
    #         bias = torch.zeros_like(bias)
    #         # Update the weights and bias
    #         if isinstance(m, nn.Conv2d):
    #             m.weight = nn.Parameter(weight, requires_grad=train_kernel)
    #             m.bias = nn.Parameter(bias, requires_grad=train_bias)
    #         else:
    #             m.weight = nn.Parameter(weight, requires_grad=train_dense)
    #             m.bias = nn.Parameter(bias, requires_grad=train_dense)
    #         if i == len(module_list) - 1:
    #             m.requires_grad = train_last

    # Get a list of all the parameters of the model
    parameters = model.parameters()

    # Get the weights and biases of the model
    weights = []
    biases = []
    for p in parameters:
        if "weight" in p.name:
            weights.append(p)
        elif "bias" in p.name:
            biases.append(p)

    # Get the L1 norm of the weights
    weight_l1 = torch.cat([torch.norm(w, p=1) for w in weights])
    # Get the 99% quantile of the L1 norm
    quantile_value = torch.quantile(weight_l1, quantile)
    # Get the indices of the weights that are less than the 99% quantile
    indices = weight_l1 < quantile_value
    # Set the weights to zero
    for i in indices:
        weights[i] = nn.Parameter(torch.zeros_like(weights[i]), requires_grad=train_weight)
    # Set the biases to zero
    for b in biases:
        b = nn.Parameter(torch.zeros_like(b), requires_grad=train_bias)
    # Update the weights and biases of the model
    # for i, w in enumerate(weights):
    #     # Freeze the weigths of the 99% quantile
    #     if i in indices:
    #         w = nn.Parameter(w, requires_grad=train_weight)
    #     else:
    #         w = nn.Parameter(w, requires_grad=True)
    # for b in biases:
    #     b = nn.Parameter(b, requires_grad=train_bias)
    
    return model
    




class ForgetLoss(nn.Module):
    def __init__(self, class_to_forget, target_format, loss_fn, classes, unlearning_type="skip", unlearning_method="confuse_vision", original=None):
        super(ForgetLoss, self).__init__()
        self.target_format = target_format
        self.classes = classes

        assert unlearning_method in ["confuse_vision", "sebastian_unlearn"], f"UNKNOWN UNLEARNING METHOD: '{unlearning_method}' must be one of the following: 'confuse_vision', 'sebastian_unlearn'"
        self.unlearning_method = unlearning_method

        #Assert original is a torch model
        assert isinstance(original, torch.nn.Module) if unlearning_method == "sebastian_unlearn" else True, f"ORIGINAL: '{original}' must be a torch model"                                                  
        self.original = original
        #Freeze original model
        for param in self.original.parameters():
            param.requires_grad = False

        if unlearning_type not in ["skip", "zero"]:
            raise Exception(f"UNKNOWN UNLEARNING TYPE: '{unlearning_type}' must be one of the following: 'skip', 'zero'")
        self.unlearning_type = unlearning_type

        if loss_fn == "huber":
            self.loss_fn = torch.nn.HuberLoss(delta=2.0)
        elif loss_fn == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif loss_fn == "mse":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise Exception(f"UNKNOWN LOSS: '{loss_fn}' must be one of the following: 'l1', 'mse', 'huber' ")     

        if "counts" in self.target_format:
            if "multilabel" in self.target_format:
                self.class_to_forget = class_to_forget
                self.class_to_forget_idx = [i for i, c in enumerate(self.classes) if c == self.class_to_forget]
                if len(self.class_to_forget_idx) == 0:
                    raise Exception(f"CLASS TO FORGET: '{class_to_forget}' must be one of the following: {self.classes}")
                print(f"Class to forget: {self.class_to_forget}")  
            else:
                raise Exception(f"MULTILABEL: 'multilabel' must be in the target format")
        else:
            raise Exception(f"COUNTS: 'counts' must be in the target format")                                                            

    def forward(self, inputs, targets):
        if "counts" in self.target_format:
            if "multilabel" in self.target_format:
                if self.unlearning_type == "skip":
                    #Skip class to forget
                    targets = torch.cat((targets[:,:self.class_to_forget_idx[0]], targets[:,self.class_to_forget_idx[0]+1:]), dim=1)
                    inputs = torch.cat((inputs[:,:self.class_to_forget_idx[0]], inputs[:,self.class_to_forget_idx[0]+1:]), dim=1)
                    #Assert that the class was removed
                    assert targets.shape[1] == inputs.shape[1] == len(self.classes) - 1, f"Class to forget was not removed: {targets.shape[1]} != {len(self.classes) - 1}"
                elif self.unlearning_type == "zero":
                    #Zero class to forget
                    targets[:,self.class_to_forget_idx[0]] = 0
                    inputs[:,self.class_to_forget_idx[0]] = 0
                    #Assert that the class was zeroed
                    assert targets[:,self.class_to_forget_idx[0]].sum() == inputs[:,self.class_to_forget_idx[0]].sum() == 0, f"Class to forget was not zeroed"
                else:
                    raise Exception(f"UNKNOWN UNLEARNING TYPE: '{self.unlearning_type}' must be one of the following: 'skip', 'zero'")
                
                if self.unlearning_method == "confuse_vision":
                    #Compute loss
                    return self.loss_fn(inputs, targets)
                elif self.unlearning_method == "sebastian_unlearn":
                    #Add MSE of entropy regularization
                    regularizer = torch.nn.MSELoss()
                    return self.loss_fn(inputs, targets) + regularizer(targets, self.original(inputs))

                else:
                    raise Exception(f"UNKNOWN UNLEARNING METHOD: '{self.unlearning_method}' must be one of the following: 'confuse_vision', 'sebastian_unlearn'")
        