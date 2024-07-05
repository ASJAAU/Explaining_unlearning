import argparse
import timm
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
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
    parser.add_argument("--heatmap_only", action="store_true", help="Enable only saving heatmap")
    parser.add_argument("--show", action="store_true", default=False, help="Enable rendering")
    parser.add_argument("--verbose", default=False, action='store_true', help="Enable verbose status printing")
    args = parser.parse_args()        

    print("\n########## classify-EXPLAIN-remove ##########")
    #Load configs
    cfg = get_config(args.config)

    #Get valid input files
    input_files, gts = get_valid_files(args.input)

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
    for i, img_path in tqdm(enumerate(input_files)):
        #Load image
        img = read_image(img_path)
        input = valid_transforms(img)
        input = input.to(args.device).unsqueeze(0)

        #get groundtruths (if possible)
        gt = gts[i]

        #initial prediction
        orig_pred = model(input)

        #Explain
        salient_map = sidu(model, model.layer4[2].act3, input, args.device)

        #visualize
        img = visualize_prediction(img, orig_pred, gt, [salient_map], blocking=args.show)

        #show?
        if args.show:
            img.show(block=True)

        #Save output
        outpath = os.path.join(args.output, os.path.dirname(img_path))
        existsfolder(outpath)
        
        if args.heatmap_only: #heatmaps only
            output_filename = os.path.splitext(img_path)[0] + ".npy"
            np.save(output_filename, salient_map)
        else: # visualize figure
            output_filename = os.path.basename(img_path)
            img.savefig(outpath + output_filename)