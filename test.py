import argparse
from datetime import datetime
import tqdm
import yaml
import numpy as np

from utils.dataloader import REPAIHarborfrontDataset
from utils.utils import existsfolder, get_config, Logger

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as torch_transforms
import timm


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser("Test XAI-MU model")
    # Positionals
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    parser.add_argument("weights", type=str, help="Path to the model weight file")
    # Optional
    parser.add_argument("--split", type=str, default="test", help="which datasplit from the config to use")
    parser.add_argument("--overide_root", type=str, default=None, help="Provide another root path for the dataset than specified in the config")
    parser.add_argument("--overide_csv", type=str, default=None, help="Provide an path to another CSV than specified in the config")
    parser.add_argument("--device", default="cuda:0", help="Which device to prioritize")
    parser.add_argument("--output", default="./assets/test/", help="Where to save the evaluation ouputs if any")
    parser.add_argument("--verbose", default=False, action='store_true', help="Enable verbose status printing")
    args = parser.parse_args()        

    print("\n########## COUNT-EXPLAIN-REMOVE ##########")
    # Load configs
    cfg = get_config(args.config)

    # Setup preprocessing steps
    test_transforms = torch_transforms.Compose([
        torch_transforms.ToDtype(torch.float32, scale=True),
        torch_transforms.ToTensor(),
    ])

    print("\n########## PREPARING DATA ##########")
    print("\n### CREATING TEST DATASET")

    #Check overide to data csv
    if args.overide_csv is not None:
        print(f"OVERIDING DATASET CSV: {args.overide_csv}")
        dataset_csv = args.overide_csv
    else:
        print(f"USING DATASET ROOT SPECIFIED BY CONFIG: {cfg['data'][args.split]}")
        dataset_csv = cfg["data"][args.split]

    #check overide to data root
    if args.overide_root is not None:
        print(f"OVERIDING DATASET ROOT: {args.overide_root}")
        dataset_root = args.overide_root
    else:
        print(f"USING DATASET SPECIFIED BY CONFIG: {cfg['data']['root']}")
        dataset_root = cfg["data"]["root"]


    # initialize training dataset
    test_dataset = REPAIHarborfrontDataset(
        data_split=dataset_csv,
        root=dataset_root,
        classes=cfg["data"]["classes"],
        transform=test_transforms,
        target_format=cfg["data"]["target_format"],
        device=args.device,
        verbose=args.verbose, #Print status and overview
    )
    
    # initialize training dataloader
    print("Creating test dataloader:")
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=cfg["training"]["batch_size"], 
        shuffle=True
        )
    
    # print example batch (sanity check)
    dummy_sample = next(iter(test_dataloader))
    print(f"Input Tensor = {dummy_sample[0].shape}")
    print(f"Label Tensor = {dummy_sample[1].shape}")
    

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
    except:
        raise Exception(f"Failed to load weights from '{args.weights}'")

    # Create / check output folder exists
    existsfolder(args.output)

    # Establish logging / eval
    if 'counts' in cfg["data"]["target_format"]:
        if 'multilabel' in cfg["data"]["target_format"]:
            logger = Logger(cfg, out_folder=args.output, metrics=cfg["evaluation"]["metrics"], classwise_metrics=cfg["data"]["classes"])
        else:
            logger = Logger(cfg, out_folder=args.output, metrics=cfg["evaluation"]["metrics"])
    elif 'binary' in cfg["data"]["target_format"]:
        if 'multilabel' in cfg["data"]["target_format"]:
            raise NotImplemented
            #logger = Logger(cfg, out_folder=None, metrics=cfg["evaluation"]["metrics"], classwise_metrics=cfg["data"]["classes"])
        else:
            raise NotImplemented
            #logger = Logger(cfg, out_folder=None, metrics=cfg["evaluation"]["metrics"])

    if cfg["wandb"]["plotting"]:
        extra_plots = {}
        from utils.wandb_plots import conf_matrix_plot
        from functools import partial
        extra_plots[f"conf_plot"] = conf_matrix_plot
        if 'multilabel' in cfg["data"]["target_format"]:
            for i,c in enumerate(cfg["data"]["classes"]):
                extra_plots[f"conf_plot_{c}"] = partial(conf_matrix_plot, idx=i)

    #Evaluate
    model.eval()

    print("\n########## RUNNING INFERENCE ##########")
    # Process all data in set
    for i, batch in tqdm.tqdm(enumerate(test_dataloader), unit="Batch", desc="Inference", leave=False, total=len(test_dataloader)):
        
        # Seperate batch
        inputs, labels = batch
        
        # Forward
        outputs = model(inputs)

        # Save predictions
        logger.add_prediction(outputs.detach().to("cpu").numpy(), labels.detach().to("cpu").numpy())

    print("########## CALCULATING METRICS ##########")
    # Check for loggin frequency
    logs = logger.log(
        clear_buffer=True,
        prepend='test',
        extras=extra_plots,
        xargs={},
    )

    print("########## RESULT DICT ##########")
    for k,v in logs.items():
        print(f"{k:>25}: {v}")

    # Highlight that results are available at url
    if cfg["wandb"]["enabled"]:
        print(f"Logs, metrics and figures available at: {logger.wandb.get_url()}")