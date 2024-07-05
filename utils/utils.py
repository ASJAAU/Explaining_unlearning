import os
import yaml
import numpy as np
from functools import partial
import utils.metrics
import glob
import pandas as pd

### GENERAL IO
def existsfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_valid_files(inputs):
    __EXT__ = (".jpg", ".png", ".bmp")
    images = []
    gts = []
    for input in inputs:
        if os.path.isfile(input) and input.endswith(__EXT__):
            images.append(input)
            gts.append(None)
        elif os.path.isfile(input) and input.endswith('.csv'):
            split = pd.read_csv(input, sep=";")
            images.extend(split["file_name"].to_list())
            gts.extend(split["label"].to_list())
        elif os.path.isdir(input):
            for ext in __EXT__:
                valid_files_in_dir=glob.glob(input + "/*" + ext)
                images.extend(valid_files_in_dir)
                gts.extend([None] * len(valid_files_in_dir))
        else:
            print(f"Invalid input: '{input}'. Skipping")

    return images, gts
            
### CONFIG FILE PARSING
def get_config(path, verbose=False):
    with open (path, 'r') as f:
        cfg = yaml.safe_load(f,)
        #If there is a base config
        if os.path.isfile(cfg["base"]):
            print(f"### LOADING BASE CONFIG PARAMETERS ({cfg['base']}) ####")
            with open (cfg["base"], 'r') as g:
                cfg = update_config(yaml.safe_load(g), cfg)
        else:
            print(f"NO CONFIG BASE DETECTED: Loading '{path}' as is")

    if verbose:
        print(yaml.dump(cfg))
    
    return cfg

def update_config(base_config, updates):
    new_config = base_config
    for key, value in updates.items():
        if type(value) == dict:
            new_config[key] = update_config(new_config[key], value)
        else:
            new_config[key] = value
    return new_config

### LOGGING
class Logger:
    def __init__(self, cfg, out_folder, metrics=[], classwise_metrics=[]) -> None:
        #Save local copies of relevant variables
        self.metrics = {}
        #Retrieve metrics
        for metric in metrics:
            self.metrics.update(get_metric(metric, classwise_metrics))
        #Establish output files and classes
        self.output_path = out_folder
        self.classes = cfg["data"]["classes"]

        #Init wandb
        if cfg["wandb"]["enabled"]:
            self.wandb = wandb_logger(
                cfg,
                output_path=out_folder,
                )
        else:
            self.wandb = None

        #Buffer for predictions
        self.preds = []
        self.labels= []

    def clear_buffer(self):
        self.preds = []
        self.labels= []

    def log(self, xargs={}, clear_buffer=True, prepend='', extras=None):
        preds = np.concatenate(self.preds, axis=0)
        labels= np.concatenate(self.labels, axis=0)

        # Create step log
        to_log = {}

        #Add xargs to log
        for k,v in xargs.items():
            to_log[k]=v

        # Apply metrics
        for name, fn in self.metrics.items():
            to_log[f'{prepend}_{name}'] = fn(preds, labels)

        #Optional Extra metrics functions (For plots etc.)
        if extras is not None:
            for name, fn in extras.items():
                to_log[f'{prepend}_{name}'] = fn(preds, labels)

        #upload to wandb
        if self.wandb is not None:
            self.wandb.log(to_log)

        #Clear buffer post logging
        if clear_buffer:
            self.clear_buffer()

        #return results
        return to_log
    
    def add_prediction(self, prediction, label):
        self.preds.append(prediction)
        self.labels.append(label)

def wandb_logger(cfg, output_path="./wandb"):
    import wandb
    if cfg["wandb"]["enabled"]:
        if cfg["wandb"]["resume"] is not None:
            wandb_logger = wandb.init(
                project=cfg["wandb"]["project_name"],
                config=cfg,
                tags=cfg["wandb"]["tags"],
                resume="must",
                id=cfg["wandb"]["resume"],
                dir=output_path,
                entity=cfg["wandb"]["entity"],
            )
        else:
            wandb_logger = wandb.init(
                project=cfg["wandb"]["project_name"],
                config=cfg,
                tags=cfg["wandb"]["tags"],
                dir=output_path,
                entity=cfg["wandb"]["entity"],
            )
        return wandb_logger
    else:
        return None
    
### HELPER FUNCTIONS FOR LOGGING
def get_metric(metric, classwise=[]):
    metrics = {}
    #MAE
    if metric == "mae":
        metrics["MAE"] = utils.metrics.mae
        for i,c in enumerate(classwise):
            metrics[f"MAE_{c}"] = partial(utils.metrics.mae, idx=i)
    #RMSE
    elif metric == "rmse":
        metrics["RMSE"] = utils.metrics.rmse
        for i,c in enumerate(classwise):
            metrics[f"RMSE_{c}"] = partial(utils.metrics.rmse, idx=i)
    #MAPE
    elif metric == "mape":
        metrics["MAPE"] = utils.metrics.mape
        for i,c in enumerate(classwise):
            metrics[f"MAPE_{c}"] = partial(utils.metrics.mape, idx=i)
    #MASE    
    elif metric == "maSe":
        metrics["MASE"] = utils.metrics.mase
        for i,c in enumerate(classwise):
            metrics[f"MASE_{c}"] = partial(utils.metrics.mase, idx=i)
    else:
        print(f"UNRECOGNIZED METRIC: {metric}, IGNORED")
    return metrics
    
def get_wandb_plots(names):
    import utils.wandb_plots as wandplots
    plots = {}
    for name in names:
        plots[name] = getattr(wandplots)
    return plots
