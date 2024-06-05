import os
import yaml

def existsfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_config(path):
    with open (path, 'r') as f:
        cfg = yaml.safe_load(f)
        #If there is a base config
        if os.path.isfile(cfg["base"]):
            print(f"### LOADING BASE CONFIG PARAMETERS ({cfg['base']}) ####")
            with open (cfg["base"], 'r') as g:
                base = yaml.safe_load(g)
                base.update(cfg)
                cfg = base
        else:
            print(f"NO CONFIG BASE DETECTED: Loading '{path}' as is")
    return cfg