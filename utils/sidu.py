# Based on :
# Pytorch SIDU: https://github.com/MarcoParola/pytorch_sidu/sidu.py
# and
# Original SIDU: https://github.com/satyamahesh84/SIDU_XAI_CODE/blob/main/SIDU_XAI.py

import torch
import torchvision
import numpy as np
from tqdm import tqdm

def kernel(vector: torch.Tensor, kernel_width: float = 0.25) -> torch.Tensor:
    """
    Kernel function for computing the weights of the differences.

    Args:
        vector (torch.Tensor): 
            The difference tensor.
        kernel_width (float, optional): 
            The kernel width. Defaults to 0.1.

    Returns:
        torch.Tensor: 
            The weights.
    """
    return torch.sqrt(torch.exp(-(vector ** 2) / kernel_width ** 2))

def normalize(array: torch.Tensor) -> torch.Tensor:
    r"""
    Normalize the array

    Args:
        array: torch.Tensor
            The input array

    Returns:
        normalized_array: torch.Tensor
            The normalized array
    """
    return (array - array.min()) / (array.max() - array.min() + 1e-13)

def uniqness_measure(masked_predictions: torch.Tensor) -> torch.Tensor:
    r""" Compute the uniqueness measure

    Args:
        masked_predictions: torch.Tensor
            The predicitons from masked featuremaps

    Returns:
        uniqueness: torch.Tensor
            The uniqueness measure
    """
    # Compute pairwise distances between each prediction vector
    distances = torch.cdist(masked_predictions, masked_predictions)

    # Compute sum along the last two dimensions to get uniqueness measure for each mask
    uniqueness = normalize(distances.sum(dim=-1))

    return uniqueness

def similarity_differences(orig_predictions: torch.Tensor, masked_predictions: torch.Tensor):
    r""" Compute the similarity differences

    Args:
        orig_predictions: torch.Tensor
            The original predictions
        masked_predictions: torch.Tensor
            The masked predictions

    Returns:
         : torch.Tensor
            The weights
        diff: torch.Tensor
            The differences
    """
    diff = abs(masked_predictions - orig_predictions).mean(axis=1)
    weights = kernel(diff)
    return weights, diff

def generate_masks(img_size: tuple, feature_map: torch.Tensor) -> torch.Tensor:
    r""" Generate masks from the feature map

    Args:
        img_size: tuple
            The size of the input image [H,W]
        feature_map: torch.Tensor
            The feature map from the model [C,H,W]

    Returns:
        masks: torch.Tensor
            The generated masks
    """

    grid = feature_map.cpu().detach().clone()
    N = feature_map.shape[0]

    #Masks placeholder of N masks
    masks = torch.Tensor(np.empty((N,*img_size))).to(feature_map.device).float()
    
    #Iterate through each mask
    for i in tqdm(range(N), desc="Generating masks"):
        #Access specific channel of featuremap
        features = feature_map[i,:,:]
        #Convert to Binary mask
        features = (features > 0.15).float()
        #Upsample the binary mask using Bi-linear interpolation
        masks[i,:,:] = torch.nn.functional.interpolate(features.unsqueeze(0).unsqueeze(0), size=(img_size), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
    return masks, grid

def get_intermediate_output(name):
    activation = {}
    def hook(model,input,output):
        activation[name] = output.detach()
    return hook

def sidu(model: torch.nn.Module, layer: torch.nn.Module, inputs: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    r""" SIDU SImilarity Difference and Uniqueness method
    Note: The original implementation processes B,H,W,C as per TF standard, where Torch uses BCHW
    
    Args:
        model: torch.nn.Module
            The model to be explained 
        image: torch.Tensor
            The input image to be explained 
        layer_name: str
            The layer name of the model to be explained
            It must be contained in named_modules() of the model
        device: torch.device, optional
            The device to use. Defaults to torch.device("cpu")

    Returns:
        saliency_maps: torch.Tensor
            The saliency maps

    """
    
    # Storage for return values
    maps = []
    predictions = []

    #Hook to catch intermediate representation
    intermediates = {}
    def hook(module, input, output):
        intermediates["repr"]=output
    
    #Connect Hook (To identify module to connect manually inspect model (i.e. print(model))
    layer.register_forward_hook(hook)

    #Disable AutoGrad
    with torch.no_grad():
        #Forward pass to extract base predictions and intermediary featuremaps
        for input in inputs:
            input = input.unsqueeze(0)
            orig_predictions = model(input).squeeze(0)
            predictions.append(orig_predictions.detach().cpu().numpy())
            orig_feature_map = intermediates["repr"].detach().clone().squeeze(0)

            #Generate masks
            masks, grid, = generate_masks((input.shape[-2],input.shape[-1]), orig_feature_map)
            N = masks.shape[0]
        
            #Predictions (explain_SIDU in original TF)
            masked_predictions = []

            #Masked batches
            batch_size = 50

            #apply masks to all 3 channels (with some channel shuffle)
            masked = masks.unsqueeze(1) * input

            #Process masked predictions
            for j in tqdm(range(0,N,batch_size), desc="Explaining"):
                masked_predictions.append(model(masked[j:min(j+batch_size,N)].to(device).float()))
            
            #align predictions
            masked_predictions = torch.cat(masked_predictions, dim=0).double()

            #Compute weights and differences
            weights, diff = similarity_differences(orig_predictions,masked_predictions)

            #Compute uniqueness to infer sample uniqueness
            uniqueness = uniqness_measure(masked_predictions)
            
            #Apply weight to uniqueness
            weighted_uniqueness = uniqueness * weights

            #Generated weighted saliency map
            saliency_map = masks * weighted_uniqueness.unsqueeze(dim=-1). unsqueeze(dim=-1)

            # reduce the saliency maps to a single map by summing over the masks dimension
            saliency_map = saliency_map.sum(dim=0)
            saliency_map /= N

            #Add to list of maps
            maps.append(saliency_map.cpu().squeeze(0).numpy())

    #Return entire batch
    return predictions, maps
  
    
if __name__ == '__main__':
    import torch
    import timm
    from torchvision.transforms import v2 as torch_transforms
    from torchvision.io import read_image
    from visualize import visualize_prediction

    # Get training image transforms
    valid_transforms = torch_transforms.Compose([
        torch_transforms.ToDtype(torch.float32, scale=True),
        torch_transforms.Resize((224,224)),
        torch_transforms.ToTensor(),
    ])

    img_paths = ["H:/Work\REPAI/Explaining_unlearning/data/sidu_test.jpg"]*4

    #Load image / images    
    input = [read_image(img_path) for img_path in img_paths]

    #Preprocess images
    input = [valid_transforms(img) for img in input]
    input = torch.stack(input, dim=0)

    #Load Pretrained model
    model = timm.create_model("resnet50d", pretrained=True,)

    # Generate SIDU Heatmaps
    predictions, heatmaps = sidu(model, layer=model.layer4[2].act3, input=input)

    #Show prediction
    for i in range(input.shape[0]):
        visualize_prediction(input[i].permute(1, 2, 0).numpy(), None, None, [heatmaps[i]], classes=["All"], blocking=True)
