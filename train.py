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
    #CLI
    parser = argparse.ArgumentParser("Train XAI-MU model")
    #Positionals
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    #Optional
    parser.add_argument("--weights", default=None, help="Optional path to weights for fintuning" )
    parser.add_argument("--device", default="cuda:0", help="Which device to prioritize")
    parser.add_argument("--output", default="./assets/train/", help="Where to save the model weights")
    parser.add_argument("--verbose", default=False, action='store_true', help="Enable verbose status printing")
    args = parser.parse_args()        

    print("\n########## COUNT-EXPLAIN-REMOVE ##########")
    #Load configs
    cfg = get_config(args.config)


    print("\n########## PREPARING DATA ##########")
    #Setup preprocessing steps
    base_transforms = [
        torch_transforms.ToDtype(torch.float32, scale=True),
        torch_transforms.ToTensor(),
    ]

    #Training transformations
    if cfg["data"]["augment"]:
        base_transforms.extend([
            torch_transforms.RandomHorizontalFlip(p=0.5),
            torch_transforms.RandomRotation(5), #5 degrees +-
            torch_transforms.ColorJitter(
                brightness  = 0.15,
                contrast    = 0.2,
                saturation  = 0.0,
                hue         = 0.0,),
            torch_transforms.RandomInvert(p=0.2),
            ]
        )
        train_transforms = torch_transforms.Compose(base_transforms)
    else:
        train_transforms = torch_transforms.Compose(base_transforms)

    #Validation transformations
    valid_transforms = torch_transforms.Compose(base_transforms)

    #Label transformations
    label_transforms  = torch_transforms.Compose([torch_transforms.ToTensor()])

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
    
    #Load optional weights
    if args.weights is not None:
        try:
            model.load_state_dict(torch.load(args.weights))
            print(f"Loaded weights from '{args.weights}'")
        except:
            raise Exception(f"Failed to load weights from '{args.weights}'")

    #Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=0.9,
        weight_decay=5e-4
        )

    #Define loss
    if cfg["training"]["loss"] == "huber":
        loss_fn = torch.nn.HuberLoss(delta=2.0)
    elif cfg["training"]["loss"] == "l1":
        loss_fn = torch.nn.L1Loss()
    elif cfg["training"]["loss"] == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise Exception(f"UNKNOWN LOSS: '{cfg['training']['loss']}' must be one of the following: 'l1', 'mse', 'huber' ")

    #Create output folder
    out_folder = f'{args.output}/{cfg["model"]["exp"]}/{datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}/'
    print(f"Saving weights and logs at '{out_folder}'")
    existsfolder(out_folder)
    existsfolder(out_folder+"/weights")

    #Save copy of config
    cfg["folder_name"] = out_folder
    with open(out_folder + "/config.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    #Logging
    if 'counts' in cfg["data"]["target_format"]:
        if 'multilabel' in cfg["data"]["target_format"]:
            logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"], classwise_metrics=cfg["data"]["classes"])
            val_logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"], classwise_metrics=cfg["data"]["classes"])
        else:
            logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"])
            val_logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"])
    elif 'binary' in cfg["data"]["target_format"]:
        if 'multilabel' in cfg["data"]["target_format"]:
            raise NotImplemented
            #logger = Logger(cfg, out_folder=None, metrics=cfg["evaluation"]["metrics"], classwise_metrics=cfg["data"]["classes"])
        else:
            raise NotImplemented
            #logger = Logger(cfg, out_folder=None, metrics=cfg["evaluation"]["metrics"])
    else: 
        raise Exception(f"Logging for {cfg['data']['target_format']} is improperly retrieved")

    #Plotting for Validation
    if cfg["wandb"]["plotting"]:
        extra_plots = {}
        from utils.wandb_plots import conf_matrix_plot
        from functools import partial
        extra_plots[f"conf_plot"] = conf_matrix_plot
        if 'multilabel' in cfg["data"]["target_format"]:
            for i,c in enumerate(cfg["data"]["classes"]):
                extra_plots[f"conf_plot_{c}"] = partial(conf_matrix_plot, idx=i)

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

            #Store prediction
            logger.add_prediction(outputs.detach().to("cpu").numpy(), labels.detach().to("cpu").numpy())

            #Log training stats
            if i % cfg["training"]["log_freq"] == 0:
                logs = logger.log(
                    clear_buffer=True,
                    prepend='train',
                    xargs={
                        "loss": running_loss/len(train_dataloader)
                    },
                )
                running_loss = 0

            #Log validation stats
            if i % cfg["evaluation"]["val_freq"] == 0 or i >= len(train_dataloader)-1:
                #Proceed to Validation
                with torch.no_grad():
                    val_loss = 0
                    val_logger.clear_buffer()
                    for j, val_batch in tqdm.tqdm(enumerate(valid_dataloader), unit="Batch", desc="Validating", leave=False, total=len(valid_dataloader)):
                        #Seperate batch
                        val_inputs, val_labels = val_batch
                        
                        #Forward
                        outputs = model(val_inputs)

                        #Calculate loss
                        val_loss += loss_fn(outputs, val_labels).item()

                        #Store prediction
                        val_logger.add_prediction(outputs.detach().to("cpu").numpy(), val_labels.detach().to("cpu").numpy())

                    #Log validation metrics
                    val_logs = val_logger.log(
                        clear_buffer=True,
                        prepend='valid',
                        extras=extra_plots,
                        xargs={
                            "loss": val_loss / len(valid_dataloader)
                        },
                    )
                    val_loss = 0
                    print(val_logs)
        #Save Model
        torch.save(model.state_dict(), out_folder + "/weights/" + f'{cfg["model"]["arch"]}-{cfg["model"]["task"]}-f{cfg["model"]["arch"]}-E{epoch}.pt')
