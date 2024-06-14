import pandas as pd
import argparse
import os
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
    dataset = pd.read_csv(args.input, sep=";")
    new_dataset = dataset.copy(deep=True)

    # Augment data
    for type in args.type:
        # Duplicate underrepresented classes (Anything but human for now)
        if type == 'duplicate_rare':
            duplicate_data = dataset.loc[(dataset["bicycle"] > 0) | (dataset["motorcycle"] > 0) | (dataset["vehicle"] > 0)]
            new_dataset = pd.concat([new_dataset, duplicate_data], ignore_index=True)
            new_dataset = new_dataset.reset_index(drop=True)
        elif type == 'duplicate_nonhuman':
            duplicate_data = dataset.loc[((dataset["bicycle"] > 0) | (dataset["motorcycle"] > 0) | (dataset["vehicle"] > 0)).any() & (dataset["human"] < 0)]
            new_dataset = pd.concat([new_dataset, duplicate_data], ignore_index=True)
            new_dataset = new_dataset.reset_index(drop=True)
        elif type == 'remove_empty':
            k = 2 #Remove every Kth element
            empties = new_dataset.loc[(new_dataset["bicycle"] < 0) & (new_dataset["motorcycle"] < 0) & (new_dataset["vehicle"] < 0) & (new_dataset["human"] < 0)]
            new_dataset = new_dataset.drop(index=empties.index.to_list()[k-1::k])
            new_dataset = new_dataset.reset_index(drop=True)

    # Merging
    if args.merge is not None:
        if len(args.merge) > 1:
            print(f"Merging: [{', '.join(args.merge[1:])}] into {args.merge[0]}")
            new_dataset[args.merge[0]] = dataset[args.merge[0]].sum(axis=1)
        else:
            print(f"Merging failed: more than 1 column name must be provided, got: {', '.join(args.merge)}")

    #Reindex to avoid repeated indices from duplication
    new_dataset = new_dataset.reset_index(drop=True)

    #Save
    new_dataset.to_csv(f'{args.output}/aug_{os.path.basename(args.input)}', ";")