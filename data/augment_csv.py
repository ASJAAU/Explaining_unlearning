import pandas as pd

import json
import os
from random import sample
import pandas as pd
import yaml
import datetime
import math
import csv
import argparse
from tqdm import tqdm

from utils.utils import existsfolder

__AUG_TYPES__ = ["duplicate_rare", "duplicate_nonhuman", "remove_empty"]

if __name__ == "__main__":
    parser  = argparse.ArgumentParser("Generate COCO annotations from Harborfront txt")
    
    #REQUIRED
    parser.add_argument('input', help="Path to the subset to augment")
    parser.add_argument('output', default="./",help="Where to save the new augmentd subset")
    #OPTIONAL
    parser.add_argument("--type", type=str, nargs= '+', default=[], choices=__AUG_TYPES__, help="type of augmentation")
    parser.add_argument("--merge", type=str, nargs= '+', help="name of rows to merge, Note: The listed columns will be merged into the first column name")
    args = parser.parse_args()

    # Load splits
    dataset = pd.DataFrame(args.input)
    new_dataset = pd.DataFrame(args.input).copy(deep=True)

    # Augment data
    for type in args.type:
        # Duplicate underrepresented classes (Anything but human for now)
        if type == 'duplicate_rare':
            new_dataset.append(dataset.loc[(dataset["bicycle"] > 0) or (dataset["motorcycle"] > 0) or (dataset["vehicle"] > 0)])
        elif type == 'duplicate_nonhuman':
            new_dataset.append(dataset.loc[((dataset["bicycle"] > 0) or (dataset["motorcycle"] > 0) or (dataset["vehicle"] > 0)) and dataset["human"] <= 0])
        elif type == 'remove_empty':
            k = 2 #Remove every Kth element
            empties = new_dataset.loc[(dataset["bicycle"] > 0) and (dataset["motorcycle"] > 0) and (dataset["vehicle"] > 0) and (dataset["vehicle"] > 0)]
            new_dataset = new_dataset.drop(index=empties.Index.to_list()[k-1::k])

    # Merging
    if len(args.merge) > 0:
        if len(args.merge) > 1:
            print(f"Merging: [{', '.join(args.merge[1:])}] into {args.merge[0]}")
            new_dataset[args.merge[0]] = dataset[args.merge[0]].sum(axis=1)
        else:
            print(f"Merging failed: more than 1 column name must be provided, got: {', '.join(args.merge)}")

