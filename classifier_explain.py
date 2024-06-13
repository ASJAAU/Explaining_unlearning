import argparse
import tqdm
import timm
import torch

from torchvision.io import read_image
from torchvision.transforms import v2 as torch_transforms
from utils.utils import existsfolder, get_config, get_valid_files
from utils.sidu import *
from utils.visualize import visualize_prediction

if __name__ == "__main__":
    #CLI
    parser = argparse.ArgumentParser("Train a multi-class binary classifier ")
    #Positionals
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    parser.add_argument("weights", type=str, help="Path to the model weight file")
    parser.add_argument("input", type=str, nargs= '+', help="Path to files/folders to run inference on")
    #Optional
    parser.add_argument("--device", default="cuda:0", help="Which device to prioritize")
    parser.add_argument("--output", default="./explained/", help="Where to save image explinations")
    parser.add_argument("--verbose", default=False, action='store_true', help="Enable verbose status printing")
    args = parser.parse_args()        

    print("\n########## classify-EXPLAIN-remove ##########")
    #Load configs
    cfg = get_config(args.config)

    #Get valid input files
    input_files = get_valid_files(args.input)

    # Get training image transforms
    valid_transforms = torch_transforms.Compose([
        torch_transforms.ToDtype(torch.float32, scale=True),
        torch_transforms.ToTensor(),
    ])
    
    print("\n########## BUILDING MODEL ##########")
    #Load model
    num_cls = 1 if 'multilabel' not in cfg["data"]["target_format"] else len(cfg["data"]["classes"])
    model = timm.create_model(
            cfg["model"]["arch"], 
            pretrained=False, 
            in_chans=1, 
            num_classes = num_cls,
            ).to(args.device)

    #Load model weights
    model.load_state_dict(torch.load(args.weights))

    #Prime model
    model.eval()
    model.to(args.device)

    #Create output folder
    existsfolder(f'{args.output}')

    print("\n########## EXPLAINING MODEL ##########")
    for img_path in tqdm.tqdm(input_files):
        #Load image
        img = read_image(img_path)
        input = valid_transforms(img)
        input = input.to(args.device)

        #initial prediction
        orig_pred = model(input)

        #Explain
        salient_map = sidu(model, model.layer4[2].act3, input, args.device)

        #visualize
        img = visualize_prediction(img, orig_pred, None, [salient_map])