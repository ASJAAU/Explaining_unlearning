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
from utils.visualize import visualize_prediction, visualize_heatmap

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
    parser.add_argument("--heatmap", action="store_true", help="Enable only saving heatmap")
    parser.add_argument("--visualization", action="store_true", help="Enable Matplotlib Visualization")
    parser.add_argument("--show", action="store_true", default=False, help="Enable screen rendering")
    parser.add_argument("--colormap", default='jet', help="Which Matplotlib Colormap to use")
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
    batch_size = 32
    for i in tqdm(range(0, len(input_files), batch_size)):
        #Load images
        file_paths = input_files[i:min(i+batch_size,len(input_files))] 
        imgs = [read_image(img_path) for img_path in file_paths]
        input = [valid_transforms(img) for img in imgs]
        input = torch.stack(input, dim=0)
        input = input.to(args.device).unsqueeze(0)

        #get groundtruths (if possible)
        gt = gts[i:min(i+batch_size,len(input_files))]

        #initial prediction
        orig_preds = model(input)

        #Explain
        salient_map = sidu(model, model.layer4[2].act3, input, args.device)

        #Create output folder
        outpath = os.path.join(args.output, os.path.dirname(file_paths[j]))
        existsfolder(outpath)

        #visualize
        if args.visualization:
            for j in range(len(imgs)):
                im = visualize_prediction(imgs[j], orig_preds[j], gt[j], [salient_map[j]], blocking=args.show, cmap=args.colormap)
                output_filename = os.path.basename(file_paths[j])
                im.savefig(outpath + output_filename)

            #show?
            if args.show:
                im.show(block=True)

        if args.heatmap: #heatmaps only
            for j in range(len(salient_map)):
                im = visualize_heatmap(imgs[j], salient_map[j], cmap=args.colormap)
                output_filename = os.path.splitext(file_paths[j])[0] + ".npy"
                np.save(outpath + output_filename, salient_map)