import os
import yaml

def existsfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_config(path, verbose=False):
    with open (path, 'r') as f:
        cfg = yaml.safe_load(f)
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



    return