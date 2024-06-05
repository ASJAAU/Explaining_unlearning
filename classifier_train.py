import argparse
from datetime import datetime
import tqdm

from models.resnet import *
from data.dataloader import REPAIHarborfrontDataset
from utils.metrics import Logger, get_metrics
from utils.saving import existsfolder, get_config
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

    print("\n########## LOADING DATA ##########")
    train_dataset = REPAIHarborfrontDataset(
        data_split=cfg["data"]["train"],
        root=cfg["data"]["root"],
        classes=cfg["data"]["classes"],
        transform=train_transforms,
        target_format=cfg["data"]["target_format"],
        verbose=True, #Print status and overview
        )

    valid_dataset = REPAIHarborfrontDataset(
        data_split=cfg["data"]["valid"],
        root=cfg["data"]["root"],
        classes=cfg["data"]["classes"],
        transform=valid_transforms,
        target_format=cfg["data"]["target_format"],
        verbose=True, #Print status and overview
        )
    
    print("Creating training dataloader:")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg["training"]["batch_size"], 
        shuffle=True
        )
    
    print("Creating validation dataloader:")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["training"]["batch_size"]
        )
    
    #Define Model
    print("\n########## BUILDING MODEL ##########")
    model = timm.create_model(
            'resnet50d', 
            pretrained=False, 
            in_chans=1, 
            num_classes = len(cfg["data"]["classes"]),
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
    loss_fn = torch.nn.BCEWithLogitsLoss()

    #Retrieve metrics for logging 
    metrics = get_metrics(cfg["data"]["target_format"])

    #Create output folder
    out_folder = f'{args.output}/{cfg["model"]["name"]}/{cfg["model"]["exp"]}_{datetime.now().strftime("%d-%m-%Y:%H")}'
    print(f"Saving weights and logs at '{out_folder}'")
    existsfolder(out_folder)
    existsfolder(out_folder+"/weights")

    # Logging
    logger = Logger(cfg, out_folder=out_folder)

    print("\n########## TRAINING MODEL ##########")
    for epoch in tqdm.tqdm(range(cfg["training"]["epochs"]), unit="Epoch", desc="Epochs"):
        #Train
        model.train()
        running_loss = 0
        for i, batch in tqdm.tqdm(enumerate(train_dataloader), unit="Batch", desc="Training", leave=False):
            #Reset gradients (redundant but sanity check)
            optimizer.zero_grad()
            
            #Seperate batch
            inputs, labels = batch
            print(inputs, labels)
            
            #Forward
            outputs = model(inputs)
            
            #Calculate loss
            loss = loss_fn(outputs, labels)
            loss.backward()
            running_loss += loss.item()

            #Propogate error
            optimizer.step()

            #logger
            logger.add_predictions(outputs.to("cpu").numpy(), labels.to("cpu").numpy())

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
        
        #Step learning rate
        lr_schedule.step()

        #Validate
        model.eval()
        running_loss = 0
        for i, batch in tqdm.tqdm(enumerate(valid_dataloader), unit="Batch", desc="Validating", leave=False):
            #Seperate batch
            inputs, labels = batch
            print(inputs, labels)
            
            #Forward
            outputs = model(inputs)

            #Calculate loss
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() / len(valid_dataloader)

            #Log metrics
            logger.add_predictions(outputs.to("cpu").numpy(), labels.to("cpu").numpy())

        #Check for loggin frequency
        logger.clear_buffer()
        logs = logger.log(
            xargs={
                "loss": running_loss
            },
            clear_buffer=True,
            prepend='valid'
        )
        print(logs)
