import argparse
from datetime import datetime
import tqdm
import yaml
import numpy as np

from data.dataloader import REPAIHarborfrontDataset
from utils.utils import Logger, get_metric
from utils.utils import existsfolder, get_config
from utils.unlearning import confuse_vision, sebastian_unlearn, ForgetLoss

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
    
    #Print model summary
    # print(model)

    #Load weights 
    try:
        model.load_state_dict(torch.load(args.weights))
        print(f"Loaded weights from '{args.weights}'")

        #Print model summary
        # print(model)

    except:
        raise Exception(f"Failed to load weights from '{args.weights}'")

    # Confuse vision (add Gaussian noise to convolutional kernels)
    print("\n########## UNLEARNING ##########")
    if cfg["unlearning"]["method"] == "confuse_vision":
        unlearned_model = confuse_vision(torch.clone(model), 
                    noise_scale = cfg["unlearning"]["noise_scale"], 
                    add_noise=cfg["unlearning"]["add_noise"],
                    trans = cfg["unlearning"]["transpose"], 
                    reinit_last = cfg["unlearning"]["reinit_last"],
                    train_dense = cfg["unlearning"]["train_dense"],
                    train_kernel = cfg["unlearning"]["train_kernel"],
                    train_bias = cfg["unlearning"]["train_bias"],
                    train_last = cfg["unlearning"]["train_last"],                
                    )
    elif cfg["unlearning"]["method"] == "sebastian_unlearn":
        unlearned_model = sebastian_unlearn(torch.clone(model),
                    train_weight=cfg["unlearning"]["train_kernel"],
                    train_bias=cfg["unlearning"]["train_bias"],
                    )
    else:
        raise Exception(f"Unlearning method '{cfg['unlearning']['method']}' not recognized")
        
    # print(model)

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
    print("\n### INITIALIZING LOSS")
    loss_fn = ForgetLoss(
        class_to_forget=cfg["unlearning"]["class_to_forget"],
        target_format=cfg["data"]["target_format"],
        loss_fn=cfg["training"]["loss"],
        classes=cfg["data"]["classes"],
        unlearning_type=cfg["unlearning"]["type"],
        unlearning_method=cfg["unlearning"]["method"],
        original=model
    )


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
            clear_buffer=True,
            prepend='valid',
            extras=extra_plots,
            xargs={
                "loss": running_loss / len(valid_dataloader)
            },
        )

        #Save Model
        torch.save(model.state_dict(), out_folder + "/weights/" + f'{cfg["model"]["arch"]}-{cfg["model"]["task"]}-Epoch{epoch}.pt')
