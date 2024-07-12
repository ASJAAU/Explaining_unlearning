import argparse
import timm
import torch
import os
import numpy as np
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as torch_transforms

from utils.utils import existsfolder, get_config, get_valid_files, Logger
from utils.sidu import *
from utils.visualize import visualize_prediction, visualize_heatmap
from data.dataloader import REPAIHarborfrontDataset

if __name__ == "__main__":
    #CLI
    parser = argparse.ArgumentParser("Train a multi-class binary classifier ")
    #Positionals
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    parser.add_argument("weights", type=str, help="Path to the model weight file")
    #Optional
    parser.add_argument("--split", default="test", help="Which datasplit to use", choices=["train","test","valid"])
    parser.add_argument("--overide_csv", type=str, default=None, help="Provide an path to another CSV than specified in the config")
    parser.add_argument("--device", default="cuda:0", help="Which device to prioritize")
    parser.add_argument("--output", default="./explained/", help="Where to save image explinations")
    parser.add_argument("--heatmap", action="store_true", help="Enable only saving heatmap")
    parser.add_argument("--visualization", action="store_true", help="Enable Matplotlib Visualization")
    parser.add_argument("--mask", action="store_true", help="Calculate heatmap coverage")
    parser.add_argument("--colormap", default='jet', help="Which Matplotlib Colormap to use")
    parser.add_argument("--show", action="store_true", help="Enable rendering the visualization on screen")
    parser.add_argument("--verbose", default=False, action='store_true', help="Enable verbose status printing")
    args = parser.parse_args()        

    print("\n########## CLASSIFY-EXPLAIN-REMOVE ##########")
    # Load configs
    cfg = get_config(args.config)

    # Setup preprocessing steps
    test_transforms = torch_transforms.Compose([
        torch_transforms.ToDtype(torch.float32, scale=True),
        torch_transforms.ToTensor(),
    ])

    print("\n########## PREPARING DATA ##########")
    print("\n### LOADING TEST DATASET")
    if args.overide_csv is not None:
        print(f"OVERIDING DATASET CSV: {args.overide_csv}")
        dataset_csv = args.overide_csv
    else:
        print(f"USING DATASET SPECIFIED BY CONFIG: {cfg["data"][args.split]}")
        dataset_csv = cfg["data"][args.split]

    # initialize training dataset
    test_dataset = REPAIHarborfrontDataset(
        data_split=dataset_csv,
        root=cfg["data"]["root"],
        classes=cfg["data"]["classes"],
        transform=test_transforms,
        target_format=cfg["data"]["target_format"],
        device=args.device,
        verbose=args.verbose, #Print status and overview
        )
    

    print("\n########## BUILDING MODEL ##########")
    # Define length of output vector
    num_cls = 1 if 'multilabel' not in cfg["data"]["target_format"] else len(cfg["data"]["classes"])
    model = timm.create_model(
            cfg["model"]["arch"], 
            pretrained=False, 
            in_chans=1, 
            num_classes = num_cls,
            ).to(args.device)
    
    try:
        model.load_state_dict(torch.load(args.weights))
        print(f"Loaded weights from '{args.weights}'")

        #Print model summary
        # print(model)
    except:
        raise Exception(f"Failed to load weights from '{args.weights}'")
    
    # Create / check output folder exists
    existsfolder(args.output)
    input_files = list(test_dataset.images)
    gts = list(test_dataset.labels)

    print("\n########## EXPLAINING MODEL ##########")
    batch_size = 2
    for i in tqdm(range(0, len(input_files), batch_size)):
        #Load images
        file_paths = input_files[i:min(i+batch_size,len(input_files))] 
        imgs = [read_image(img_path) for img_path in file_paths]
        input = [test_transforms(img) for img in imgs]
        input = torch.stack(input, dim=0)
        input = input.to(args.device)

        #get groundtruths (if possible)
        gt = gts[i:min(i+batch_size,len(input_files))]

        #Explain
        predictions, salient_map = sidu(model, model.layer4[2].act3, input, args.device)
        for j in range(len(imgs)):
            #visualize
            if args.visualization:
                outpath = os.path.join(args.output + "/visualize/", os.path.dirname(file_paths[j].replace(test_dataset.root, "")))
                existsfolder(outpath)
                im = visualize_prediction(imgs[j].squeeze(0), predictions[j], gt[j], [salient_map[j]], cmap=args.colormap)
                output_filename = os.path.basename(file_paths[j])
                im.savefig(outpath + "/" + output_filename)

            if args.heatmap: #heatmaps only
                outpath = os.path.join(args.output + "/heatmap/", os.path.dirname(file_paths[j].replace(test_dataset.root, "")))
                existsfolder(outpath)
                im = visualize_heatmap(imgs[j].squeeze(0), salient_map[j], cmap=args.colormap)
                output_filename = os.path.splitext(os.path.basename(file_paths[j]))[0] + ".npy"
                np.save(outpath + "/" + output_filename, salient_map)

            if args.mask:
                outpath = os.path.join(args.output+"/mask/", os.path.dirname(file_paths[j].replace(test_dataset.root, "")))
                existsfolder(outpath)
                masks = {}
                for kname in ["human", "bicycle", "vehicle", "motorcycle"]:
                    masks[kname] = np.zeros_like(imgs[j].squeeze(0)).astype(np.uint8)
                #Load annotation
                with open(os.path.join(test_dataset.root, file_paths[j].replace("frames/","annotations/").replace("image","annotations").replace("jpg","txt")), 'r') as f:
                    reader = csv.reader(f, delimiter=" ")
                    for row in reader:
                        if row != ''.join(row).strip(): #ignore empty annotations
                            #Get class name
                            cls_name = row[1]
                            #Get object boundingbox
                            bbox = [int(row[2]), int(row[3]), int(row[4]), int(row[5])]
                            #Draw boundingbox on mask
                            masks[cls_name][bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
                #save all the masks
                for key, value in masks.items():
                    output_filename = os.path.splitext(os.path.basename(file_paths[j]))[0] + f"_{key}.npy"
                    np.save(outpath + "/" + output_filename, value)
                    plt.imsave(outpath + "/" + output_filename[:-3]+"jpg", value, cmap="gray")

                #Close any MPL renderers
                plt.close('all')