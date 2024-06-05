import argparse
import yaml

from models.resnet import *
from data.dataloader import *
from torch.utils.data import Dataloader


if __name__ == "__main__":
    #CLI
    parser = argparse.ArgumentParser("Train a multi-class binary classifier ")
    #Positionals
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    #Optional
    parser.add_argument("--device", default="/GPU:1", help="Tensorflow device to prioritize", choices=["/CPU:0","/GPU:0", "/GPU:1"])
    parser.add_argument("--output", default="./assets/", help="Where to save the model weights")
    args = parser.parse_args()        

    print("\n########## CLASSIFY-EXPLAIN-REMOVE ##########")
    #Load configs
    with open (args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        #If there is a base config
        if os.path.isfile(cfg["base"]):
            print(f"### LOADING BASE CONFIG PARAMETERS ({cfg['base']}) ####")
            with open (cfg["base"], 'r') as g:
                base = yaml.safe_load(g)
                base.update(cfg)
                cfg = base
        else:
            print(f"NO CONFIG BASE DETECTED: Loading '{args.config}' as is")



    print("\n########## LOADING DATA ##########")
    train_dataset = REPAIHarborfrontDataset(
        data_split=cfg["data"]["train"],
        root=cfg["data"]["root"],
        classes=cfg["model"]["classes"],
        target_format=cfg["data"]["target_format"],
        verbose=True, #Print status and overview
        )
    print("\n########## LOADING DATA ##########")
    train_dataset = REPAIHarborfrontDataset(
        data_split=cfg["data"]["valid"],
        root=cfg["data"]["root"],
        classes=cfg["model"]["classes"],
        target_format=cfg["data"]["target_format"],
        verbose=True, #Print status and overview
        )
    
    print("Creating training dataloader:")
    train_dataloader = Dataloader(
        train_dataset, 
        batch_size=cfg["training"]["batch_size"], 
        shuffle=True
        )
    
    print("Creating validation dataloader:")
    valid_dataloader = Dataloader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"]
        )
    #Define Model
    print("\n########## BUILDING MODEL ##########")
    print(f'Building model: ResNet{cfg["model"]["size"]}')
    model = Resnet(cfg)

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
