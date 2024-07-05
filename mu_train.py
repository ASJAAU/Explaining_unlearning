import argparse
from datetime import datetime
import tqdm
import yaml
import numpy as np

from data.dataloader import REPAIHarborfrontDataset
from utils.utils import Logger, get_metric
from utils.utils import existsfolder, get_config
from utils.unlearning import confuse_vision, prune_reinit, ForgetLoss

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as torch_transforms
import timm


import torch
import torch.nn as nn
import torch.nn.functional as F



if __name__ == "__main__":
    #CLI
    parser = argparse.ArgumentParser("Perform vision confusion MU method on selected model.")
    #Positionals
    parser.add_argument("--weights", type=str, help="Path to the model weight file")
    parser.add_argument("--config", type=str, help="Path to config file (YAML)")
    #Optional
    parser.add_argument("--device", default="cuda:0", help="Which device to prioritize")
    parser.add_argument("--output", default="./assets/", help="Where to save the model weights")
    parser.add_argument("--verbose", default=False, action='store_true', help="Enable verbose status printing")
    args = parser.parse_args()        



    print("\n########## CLASSIFY-EXPLAIN-REMOVE ##########")
    #Load configs
    cfg = get_config(args.config)

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



    print("\n########## BUILDING MODEL ##########")
    #Define length of output vector
    num_cls = 1 if 'multilabel' not in cfg["data"]["target_format"] else len(cfg["data"]["classes"])
    print(num_cls)
    model = timm.create_model(
            cfg["model"]["arch"], 
            pretrained=False, 
            in_chans=1, 
            num_classes = num_cls,
            ).to(args.device)
    
    #Load weights 
    try:
        model.load_state_dict(torch.load(args.weights))
        print(f"Loaded weights from '{args.weights}'")

        #Print model summary
        # print(model)
    except:
        raise Exception(f"Failed to load weights from '{args.weights}'")

    

    print("\n########## UNLEARNING ##########")
    if cfg["unlearning"]["method"] == "confuse_vision":
        print("Confusing vision")
        model = confuse_vision(model, cfg)
    elif cfg["unlearning"]["method"] == "sebastian_unlearn":
        print("Pruning or reinitializing")
        model = prune_reinit(model, cfg)
    elif cfg["unlearning"]["method"] == "fine_tune":
        print("Fine tuning")
    else:
        raise Exception(f"Unlearning method '{cfg['unlearning']['method']}' not recognized")


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
    
    #Initialize training dataloader
    print("Creating training dataloader:")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg["training"]["batch_size"], 
        shuffle=True
        )
    
    print("\n### CREATING VALIDATION DATASET")
    #Initialize training dataset
    valid_dataset = REPAIHarborfrontDataset(
        data_split=cfg["data"]["valid"],
        root=cfg["data"]["root"],
        classes=cfg["data"]["classes"],
        transform=valid_transforms,
        target_format=cfg["data"]["target_format"],
        device=args.device,
        verbose=args.verbose, #Print status and overview
        )
    
    #Initialize validation dataloader
    print("Creating validation dataloader:")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["training"]["batch_size"]
        )


    print("\n########## INITIALIZING TRAINING ##########")
    #Define optimizer    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=cfg["training"]["lr"],
        momentum=0.9, 
        weight_decay=5e-4
        )

    #Define loss
    print("\n### INITIALIZING LOSS")
    loss_fn = ForgetLoss(cfg)

    #Retrieve metrics for logging 
    print("\n### INITIALIZING METRICS")
    metrics = get_metric(cfg["data"]["target_format"])

    #Create output folder
    out_folder = f'{args.output}/{cfg["model"]["task"]}-{cfg["model"]["arch"]}-{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
    print(f"Saving weights and logs at '{out_folder}'")
    existsfolder(out_folder)
    existsfolder(out_folder+"/weights")

    #Save copy of config
    cfg["folder_name"] = out_folder
    with open(out_folder + "/config.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    if 'counts' in cfg["data"]["target_format"]:
        if 'multilabel' in cfg["data"]["target_format"]:
            logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"], classwise_metrics=cfg["data"]["classes"])
        else:
            logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"])
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
        from utils.wandb_plots import conf_matrix, conf_matrix_plot
        from functools import partial
        #extra_plots[f"conf"] = conf_matrix
        extra_plots[f"conf_plot"] = conf_matrix_plot
        if 'multilabel' in cfg["data"]["target_format"]:
            for i,c in enumerate(cfg["data"]["classes"]):
                #extra_plots[f"conf_{c}"] = partial(conf_matrix, idx=i)
                extra_plots[f"conf_plot_{c}"] = partial(conf_matrix_plot, idx=i)



    print("\n########## TRAINING MODEL ##########")
    epochs = cfg["training"]["epochs"]
    for epoch in tqdm.tqdm(range(epochs), unit="Epoch", desc="Epochs"):
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
            loss = loss_fn(inputs, outputs, labels)
            loss.backward()
            running_loss += loss.item()

            #Propogate error
            optimizer.step()

            #logger
            logger.add_prediction(outputs.detach().to("cpu").numpy(), labels.detach().to("cpu").numpy())

            #Check for loggin frequency
            if i % cfg["training"]["log_freq"] == 0:
                logs = logger.log(
                    xargs={
                        "loss": running_loss
                    },
                    clear_buffer=True,
                    prepend='train'
                )
                running_loss = 0
                #print(logs)

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
            loss = loss_fn(inputs, outputs, labels)
            running_loss += loss.item() / len(valid_dataloader)

            #Log metrics
            logger.add_prediction(outputs.detach().to("cpu").numpy(), labels.detach().to("cpu").numpy())

        #Check for loggin frequency
        logs = logger.log(
            clear_buffer=True,
            prepend='valid',
            extras=extra_plots,
            xargs={
                "loss": running_loss / len(valid_dataloader)
            },
        )

        #Save Model
        torch.save(model.state_dict(), out_folder + "/weights/" + f'{cfg["model"]["arch"]}-{cfg["model"]["task"]}-Epoch{epoch}.pt')

        #Confuse vision again for the last 2 epochs
        if cfg["unlearning"]["method"] == "confuse_vision":
            if epoch == epochs - 2:
                cfg["unlearning"]["noise_scale"] = cfg["unlearning"]["noise_scale2"]
                cfg["unlearning"]["trans"] = False
                cfg["unlearning"]["reinit_last"] = False
                model = confuse_vision(model, cfg)
