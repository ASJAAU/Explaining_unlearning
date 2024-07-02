# Based on :
# Pytorch SIDU: https://github.com/MarcoParola/pytorch_sidu/sidu.py
# and
# Original SIDU: https://github.com/satyamahesh84/SIDU_XAI_CODE/blob/main/SIDU_XAI.py

import torch
import torchvision
import numpy as np
from tqdm import tqdm

def kernel(vector: torch.Tensor, kernel_width: float = 0.1) -> torch.Tensor:
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
    uniqueness = distances.sum(dim=-1)

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
    diff = abs(masked_predictions - orig_predictions)
    # compute the average of the differences, from (batch, num_masks, num_features, w, h) -> (batch, num_masks, w, h)
    diff = diff.mean(dim=2)
    weighted_diff = kernel(diff)
    # compute the average of the weights, from (batch, num_masks, w, h) -> (batch, num_masks)
    weighted_diff = weighted_diff.mean(dim=(2, 3))
    return weighted_diff, diff

def generate_masks(img_size: tuple, feature_map: torch.Tensor, s: int = 8) -> torch.Tensor:
    r""" Generate masks from the feature map

    Args:
        img_size: tuple
            The size of the input image [H,W]
        feature_map: torch.Tensor
            The feature map from the model [C,H,W]
        s: int
            The scale factor

    Returns:
        masks: torch.Tensor
            The generated masks
    """
    h, w = img_size
    cell_size = np.ceil(np.array(img_size) / s)

    #Rotate axis to [C,H,W]
    grid = feature_map.detach().clone()
    N = feature_map.shape[0]

    #Masks placeholder of N masks
    masks = np.empty((N,*img_size))
    
    #Iterate through each mask
    for i in tqdm(range(N), desc="Generating masks"):
        #Access specific channel of featuremap
        features = feature_map[i,:,:]
        #Convert to Binary mask
        features = (features > 0.15).float()
        #Upsample the binary mask using Bi-linear interpolation
        masks[i,:,:] = torch.nn.functional.interpolate(features.unsqueeze(0).unsqueeze(0), size=(img_size), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
    return torch.from_numpy(masks), grid, cell_size

def get_intermediate_output(name):
    activation = {}
    def hook(model,input,output):
        activation[name] = output.detach()
    return hook

def sidu(model: torch.nn.Module, layer: torch.nn.Module, image: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
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
    
    # Storage
    maps = []

    #Hook to catch intermediate representation
    intermediates = {}
    def hook(module, input, output):
        intermediates["repr"]=output
    
    #Connect Hook (To identify module to connect manually inspect model (i.e. print(model))
    layer.register_forward_hook(hook)

    #Disable AutoGrad
    with torch.no_grad():
        #Forward pass to extract base predictions and intermediary featuremaps
        orig_predictions = model(image)
        orig_feature_map = intermediates["repr"].detach().clone()

        # #Iterate over the batch dimension
        for i in range(image.shape[0]):
            #Note torch.movedim(orig_feature_map[i],0,2) is used to change to BHW, to retain original TF code structure

            #Generate masks
            masks, grid, cell_size = generate_masks((image.shape[-2],image.shape[-1]), orig_feature_map[i], s=8)
            N = masks.shape[0]

            print(masks.shape, grid.shape, cell_size)
        
            #Predictions (explain_SIDU in original TF)
            preds = []

            #Masked batches
            batch_size = 100

            print(f"Inpt image: {image.shape}")
            #apply masks to all 3 channels (with some channel shuffle)
            masked = masks.unsqueeze(1) * image
            print(f"Masks: {masked.shape}")

            #Process masked predictions
            for j in tqdm(range(0,N,batch_size), desc="Explaining"):
                preds.append(model(masked[j:min(i+batch_size,N)].to(device)))
            
            #align predictions
            preds = np.concatenate(preds)
            print(f'Preds: {preds.shape}')

            #Compute weights and differences
            weights, diff = similarity_differences(orig_predictions[i], preds)



        #     # Repeat the masks 3 times along the channel dimension to match the number of channels of the image and masks
        #     masks = masks.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        #     images = image.unsqueeze(1).repeat(1, num_masks, 1, 1, 1)
        #     masked_images = images * masks

        #     # Compute masked featuremaps
        #     masked_feature_map = []
        #     for i in range(num_masks):
        #         masked_feature_map.append(model(masked_images[:, i, :, :, :])['target_layer'])
        #     masked_feature_map = torch.stack(masked_feature_map, dim=1) # TODO speed up this part
            
        #     orig_feature_map[i] = orig_feature_map[i].unsqueeze(1).repeat(1, num_masks, 1, 1, 1)

        #     # compute the differences of the similarity and the uniqueness
        #     weighted_diff, difference = similarity_differences(orig_feature_map[i], masked_feature_map)
        #     uniqness = uniqness_measure(masked_feature_map)

        #     # compute SIDU
        #     sidu = weighted_diff * uniqness

        #     # reduce the masks size by removing the channel dimension (batch, num_masks, 3, w, h) -> (batch, num_masks, 1, w, h)
        #     masks = masks.mean(dim=2, keepdim=True)
        #     masks = masks.squeeze(2)

        #     # compute saliency maps by averaging the masks weighted by the SIDU
        #     # each mask of masks (batch, num_masks, w, h) must be multiplied by the SIDU (batch, num_masks)
        #     saliency_maps = masks * sidu.unsqueeze(2).unsqueeze(3)

        #     # reduce the saliency maps to a single map by summing over the masks dimension
        #     saliency_maps = saliency_maps.sum(dim=1)
        #     saliency_maps /= num_masks

        return maps
    
if __name__ == '__main__':
    import torch
    import timm
    from torchvision.transforms import v2 as torch_transforms
    from torchvision.io import read_image

    # Get training image transforms
    valid_transforms = torch_transforms.Compose([
        torch_transforms.ToDtype(torch.float32, scale=True),
        torch_transforms.Resize((224,224)),
        torch_transforms.ToTensor(),
    ])

    img_path = "data/sidu_test.jpg"
    #Load image
    img = read_image(img_path)
    input = valid_transforms(img)
    input = input.unsqueeze(0)

    #Load Pretrained model
    model = timm.create_model("resnet50d", pretrained=True,)

    # Generate SIDU Heatmaps
    heatmaps = sidu(model, layer=model.layer4[2].act3, image=input)