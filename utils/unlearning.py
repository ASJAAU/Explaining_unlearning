import torch
from torch import nn
from torch.nn.utils import prune

from tqdm import tqdm
from math import sqrt


def confuse_vision(model, cfg):
    """ Add Gaussian noise to the conv2d layers of the model.
        - model: a tf model with loaded weights
        - noise_scale: scale of the std of the Gaussian noise
        - trans: transpose kernel of cnn?
        - reinit_last: reinitialize last layer? otherwise add noise
    """
    noise_scale = cfg["unlearning"]["noise_scale"] 
    add_noise=cfg["unlearning"]["add_noise"]
    trans = cfg["unlearning"]["transpose"] 
    reinit_last = cfg["unlearning"]["reinit_last"]
    train_dense = cfg["unlearning"]["train_dense"]
    train_kernel = cfg["unlearning"]["train_kernel"]
    train_bias = cfg["unlearning"]["train_bias"]
    train_last = cfg["unlearning"]["train_last"]                
        
    #Get conv2d layers
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    print(f"Adding noise to {len(conv_layers)} convolutional layers")

    #Add noise to each kernel
    for i, conv in tqdm(enumerate(conv_layers), total=len(conv_layers)):
        # print(f"Convolution layer {i}: {conv}")

        #Confusing kernels
        kernel = torch.clone(conv.weight)
        for j in range(kernel.shape[0]):
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
                    # noise_std = torch.std(kernel[j,k,:,:]).item() * noise_scale
                    # # print(noise_std)
                    # if not noise_std > 0:
                    #     # print("No noise added")
                    #     noise_std = 0.01
                    noise_std = noise_scale
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
                # noise_std = torch.std(bias).item() * noise_scale
                # if not noise_std > 0:
                #     noise_std = 0.01
                noise_std = noise_scale
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
                    # noise_std = torch.std(weight).item() * noise_scale
                    # if not noise_std > 0:
                    #     noise_std = 0.01
                    noise_std = noise_scale
                    #Add weight noise
                    weight += torch.normal(mean=0, std=noise_std, size=weight.shape, device=weight.device)
                    #Update weight
                    m.weight = nn.Parameter(weight, requires_grad=train_last)

                    if m.bias:
                        print("There is bias in last layer")
                        bias = torch.clone(m.bias)
                        #Compute noise scale proportional to the std of the bias
                        # noise_std = torch.std(bias).item() * noise_scale
                        # if not noise_std > 0:
                        #     noise_std = 0.01
                        noise_std = noise_scale
                        #Add bias noise
                        bias += torch.normal(mean=0, std=noise_std, size=bias.shape, device=bias.device)
                        #Update weight and bias
                        m.bias = nn.Parameter(bias, requires_grad=train_last)
                else: 
                    print(f"Last layer is not linear: {m}")
                    raise ValueError("Last layer is not linear")
            
    return model


def prune_reinit(model, cfg):
    amount = cfg["unlearning"]["amount"]
    rand_init = cfg["unlearning"]["rand_init"]
    if rand_init:
        print("Reinitializing")
    else:
        print("Pruning")

    #Modules to prune
    modules = list()
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            modules.append((m, "weight"))
            if m.bias is not None:
                modules.append((m, "bias"))
    
    #Prune criteria
    prune.global_unstructured(
        parameters=modules,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    #Perform the pruning
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.remove(m, "weight")
            if m.bias is not None:
                prune.remove(m, "bias")
    
    #Reinitialize the pruned weights
    if rand_init: 
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                mask = m.weight == 0
                c_in = mask.shape[1]
                k = 1 / (c_in * mask.shape[2] * mask.shape[3])
                randinit = (torch.rand_like(m.weight) - 0.5) * 2 * sqrt(k)
                m.weight.data[mask] = randinit[mask]
            if isinstance(m, nn.Linear):
                mask = m.weight == 0
                c_in = mask.shape[1]
                k = 1 / c_in
                randinit = (torch.rand_like(m.weight) - 0.5) * 2 * sqrt(k)
                m.weight.data[mask] = randinit[mask]

    return model


class ForgetLoss(nn.Module):
    def __init__(self, cfg, original=None, lambda_=0.1):
        super(ForgetLoss, self).__init__()
        self.target_format=cfg["data"]["target_format"]

        unlearning_method=cfg["unlearning"]["method"]
        loss_fn=cfg["training"]["loss"]

        #Assert that the unlearning method is valid
        assert unlearning_method in ["confuse_vision", "sebastian_unlearn", "fine_tune"], f"UNKNOWN UNLEARNING METHOD: '{unlearning_method}' must be one of the following: 'confuse_vision', 'sebastian_unlearn'"
        self.unlearning_method = unlearning_method

        #Set loss function
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
                #Multi count regression
                #Assert that the unlearning type is valid
                unlearning_type=cfg["unlearning"]["type"]
                assert unlearning_type in ["skip", "zero"], f"UNKNOWN UNLEARNING TYPE: '{unlearning_type}' must be one of the following: 'skip', 'zero'"
                self.unlearning_type = unlearning_type

                #Assert original is a torch model if using sebastian_unlearn        
                assert isinstance(original, torch.nn.Module) if unlearning_method == "sebastian_unlearn" else True, f"ORIGINAL: '{original}' must be a torch model"                                                  
                self.original = original
                #Freeze original model
                for param in self.original.parameters():
                    param.requires_grad = False

                #Set class to forget
                self.classes=cfg["data"]["classes"]
                self.class_to_forget=cfg["unlearning"]["class_to_forget"]
                self.class_to_forget_idx = [i for i, c in enumerate(self.classes) if c == self.class_to_forget]
                if len(self.class_to_forget_idx) == 0:
                    raise Exception(f"CLASS TO FORGET: '{self.class_to_forget}' must be one of the following: {self.classes}")
                print(f"Class to forget: {self.class_to_forget}")  
            else:
                #Single count regression
                #No need to set class to forget
                pass
        else:
            raise Exception(f"COUNTS: 'counts' must be in the target format")                                                            

    def forward(self, outputs, labels):
        if "counts" in self.target_format:
            if "multilabel" in self.target_format:
                labels = self.set_unlearn_target(labels)
                outputs = self.set_unlearn_target(outputs)
                if self.unlearning_method == "confuse_vision":
                    #Compute loss
                    return self.loss_fn(outputs, labels)
                elif self.unlearning_method == "sebastian_unlearn":
                    #Add MSE of entropy regularization
                    regularizer = torch.nn.MSELoss()
                    return self.loss_fn(outputs, labels)
                else:
                    #Not implemented error
                    raise NotImplementedError(f"UNLEARNING METHOD: '{self.unlearning_method}' is not implemented")
            else:
                if self.unlearning_method in ["confuse_vision", "sebastian_unlearn", "fine_tune"]:
                    #Compute loss
                    return self.loss_fn(outputs, labels) 
                else:
                    #Not implemented error
                    raise NotImplementedError(f"UNLEARNING METHOD: '{self.unlearning_method}' is not implemented")

        else:
            raise Exception(f"COUNTS: 'counts' must be in the target format")
                
    def set_unlearn_target(self, tensor):
        if self.unlearning_type == "skip":
            #Skip class to forget
            tensor = torch.cat((tensor[:,:self.class_to_forget_idx[0]], tensor[:,self.class_to_forget_idx[0]+1:]), dim=1)
            #Assert that the class was removed
            assert tensor.shape[1] == len(self.classes) - 1, f"Class to forget was not removed: {tensor.shape[1]} != {len(self.classes) - 1}"
        elif self.unlearning_type == "zero":
            #Zero class to forget
            tensor[:,self.class_to_forget_idx[0]] = 0
            #Assert that the class was zeroed
            assert tensor[:,self.class_to_forget_idx[0]].sum() ==  0, f"Class to forget was not zeroed"
        else:
            raise Exception(f"UNKNOWN UNLEARNING TYPE: '{self.unlearning_type}' must be one of the following: 'skip', 'zero'")
        return tensor
    
