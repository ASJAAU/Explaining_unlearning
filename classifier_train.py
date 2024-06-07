import argparse
from datetime import datetime
import tqdm
import yaml
import numpy as np

from models.resnet import *
from data.dataloader import REPAIHarborfrontDataset
from utils.metrics import Logger, get_metrics
from utils.utils import existsfolder, get_config

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as torch_transforms
import timm


if __name__ == "__main__":
    #CLI
    parser = argparse.ArgumentParser("Train a multi-class binary classifier ")
    #Positionals
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    #Optional
    parser.add_argument("--device", default="cuda:0", help="Which device to prioritize")
    parser.add_argument("--output", default="./assets/", help="Where to save the model weights")
    parser.add_argument("--verbose", default=False, action='store_true', help="Enable verbose status printing")
    args = parser.parse_args()        

    print("\n########## CLASSIFY-EXPLAIN-REMOVE ##########")
    #Load configs
    cfg = get_config(args.config)

    #Setup preprocessing steps
    train_transforms = torch_transforms.Compose([
        torch_transforms.RandomHorizontalFlip(p=0.5),
        torch_transforms.ToDtype(torch.float32, scale=True),
        torch_transforms.ToTensor(),
    ])
    valid_transforms = torch_transforms.Compose([
        torch_transforms.ToDtype(torch.float32, scale=True),
        torch_transforms.ToTensor(),
    ])
    label_transforms  = torch_transforms.Compose([
        torch_transforms.ToTensor(),
    ])

    print("\n########## PREPARING DATA ##########")
    print("\n### CREATING TRAINING DATASET")
    #initialize training dataset
    train_dataset = REPAIHarborfrontDataset(
        data_split=cfg["data"]["train"],
        root=cfg["data"]["root"],
        classes=cfg["data"]["classes"],
        transform=train_transforms,
        target_format=cfg["data"]["target_format"],
        device=args.device,
        verbose=args.verbose, #Print status and overview
        )
    
    #initialize training dataloader
    print("Creating training dataloader:")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg["training"]["batch_size"], 
        shuffle=True
        )
    #print example batch (sanity check)
    dummy_sample = next(iter(train_dataloader))
    print(f"Input Tensor = {dummy_sample[0].shape}")
    print(f"Label Tensor = {dummy_sample[1].shape}")
    
    print("\n### CREATING VALIDATION DATASET")
    #initialize training dataset
    valid_dataset = REPAIHarborfrontDataset(
        data_split=cfg["data"]["valid"],
        root=cfg["data"]["root"],
        classes=cfg["data"]["classes"],
        transform=valid_transforms,
        target_format=cfg["data"]["target_format"],
        device=args.device,
        verbose=args.verbose, #Print status and overview
        )
    
    #initialize validation dataloader
    print("Creating validation dataloader:")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["training"]["batch_size"]
        )
    #print example batch (sanity check)
    dummy_sample = next(iter(valid_dataloader))
    print(f"Input Tensor = {dummy_sample[0].shape}")
    print(f"Label Tensor = {dummy_sample[1].shape}")

    print("\n########## BUILDING MODEL ##########")
    #Define length of output vector
    num_cls = 1 if 'multilabel' not in cfg["data"]["target_format"] else len(cfg["data"]["classes"])
    model = timm.create_model(
            cfg["model"]["arch"], 
            pretrained=False, 
            in_chans=1, 
            num_classes = num_cls,
            ).to(args.device)

    #Define optimizer
    optimizer= torch.optim.Adam(
        model.parameters(), 
        lr=cfg["training"]["lr"]
        )

    #Define learning-rate schedule
    lr_schedule = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor = 1.0, 
        end_factor = cfg["training"]["lr_decay"], 
        total_iters=cfg["training"]["epochs"],
        )
    
    #Define loss
    loss_fn = torch.nn.CrossEntropyLoss()#torch.nn.BCEWithLogitsLoss()

    #Retrieve metrics for logging 
    metrics = get_metrics(cfg["data"]["target_format"])

    #Create output folder
    out_folder = f'{args.output}/{cfg["model"]["task"]}-{cfg["model"]["arch"]}-{datetime.now().strftime("%Y_%m_%d_%H-%M")}'
    print(f"Saving weights and logs at '{out_folder}'")
    existsfolder(out_folder)
    existsfolder(out_folder+"/weights")

    #Save copy of config
    with open(out_folder + "/config.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Logging
    if 'multilabel' not in cfg["data"]["target_format"]:
        logger = Logger(cfg, out_folder=out_folder)
    elif 'multilabel' in cfg["data"]["target_format"]:
        logger = Logger(cfg, out_folder=out_folder, classwise_metrics=cfg["data"]["classes"])
    else: 
        raise Exception(f"Logging for {cfg['data']['target_format']} is improperly retrieved")

    print("\n########## TRAINING MODEL ##########")
    for epoch in tqdm.tqdm(range(cfg["training"]["epochs"]), unit="Epoch", desc="Epochs"):
        #Train
        model.train()
        running_loss = 0
        for i, batch in tqdm.tqdm(enumerate(train_dataloader), unit="Batch", desc="Training", leave=False, total=len(train_dataloader)):
            #Reset gradients (redundant but sanity check)
            optimizer.zero_grad()
            
            #Seperate batch
            inputs, labels = batch
            
            #Forward
            outputs = model(inputs)
            
            #Calculate loss
            loss = loss_fn(outputs, labels)
            loss.backward()
            running_loss += loss.item()

            #Propogate error
            optimizer.step()

            #logger
            logger.add_prediction(outputs.detach().to("cpu").numpy(), labels.detach().to("cpu").numpy())

            #Check for loggin frequency
            if i % cfg["wandb"]["log_freq"] == 0:
                logs = logger.log(
                    xargs={
                        "loss": running_loss
                    },
                    clear_buffer=True,
                    prepend='train'
                )
                running_loss = 0
                #print(logs)
        
        #Step learning rate
        lr_schedule.step()

        #Validate
        model.eval()
        running_loss = 0
        logger.clear_buffer()
        for i, batch in tqdm.tqdm(enumerate(valid_dataloader), unit="Batch", desc="Validating", leave=False, total=len(valid_dataloader)):
            #Seperate batch
            inputs, labels = batch
            
            #Forward
            outputs = model(inputs)

            #Calculate loss
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() / len(valid_dataloader)

            #Log metrics
            logger.add_prediction(outputs.detach().to("cpu").numpy(), labels.detach().to("cpu").numpy())

        #Check for loggin frequency
        logs = logger.log(
            xargs={
                "loss": running_loss / len(valid_dataloader)
            },
            clear_buffer=True,
            prepend='valid'
        )

        #Save Model
        torch.save(model.state_dict(), out_folder + "/weights/" + f'{cfg["model"]["arch"]}-{cfg["model"]["task"]}-Epoch{epoch}.pt')
