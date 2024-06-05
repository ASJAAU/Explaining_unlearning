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


def load_datasplits(path):
    with open(path) as f:
        sub_sets = yaml.load(f, Loader=yaml.FullLoader)
    splits = {}
    for s in sub_sets.keys():
        splits[s] = sub_sets[s]
    return splits

def load_annotations(sample_name, img_path, ann_path):
    #Label Storage
    uids = [] 
    labels = []
    boxes = []
    areas = []  
    ocls = []
    centers = []

    
    #Load annotations
    with open(ann_path, "r") as annotations:
        reader = csv.reader(annotations, delimiter=" ")
        for row in reader:
            if row != ''.join(row).strip(): #ignore empty annotations
                #Add unique object id                
                uids.append(int(row[0]))
                #Add class labels
                labels.append(row[1])
                #Add occlusion tags
                ocls.append(int(row[6]))
                #Add bounding box
                #boxes.append([int(row[2]), int(row[3]), int(row[4]), int(row[5])]) #Xtopleft, Ytopleft, Xbottomright, Ybottomright
                boxes.append([int(row[2]), int(row[3]), abs(int(row[4])-int(row[2])), abs(int(row[5])-int(row[3]))]) #X,Y,W,H
                #Add boundingbox Centers
                centers.append([int(row[2]), int(row[3])])
                #Add areas
                areas.append(abs(int(row[4])-int(row[2])) * abs(int(row[5])-int(row[3])))

    #Parse timestamp from samplename
    timestamp =  datetime.datetime.fromisoformat('{}-{}-{} {}:{}'.format(sample_name[:4],  #Year
                                                                        sample_name[4:6],    #Month
                                                                        sample_name[6:8],    #Day
                                                                        sample_name[-9:-7],  #hour
                                                                        sample_name[-7:-5]))  #minute
    #Save target dict
    targets = {
        "uids"       : uids,
        "boxes"     : boxes,
        "labels"    : labels,
        #"occlusions": ocls,
        "iscrowd"   : ocls, #iscrowd is used as an occlusion tag
        "areas"      : areas,
        "centers"    : centers,
        "timestamp" : timestamp + datetime.timedelta(0, int(sample_name[-4:]))
        #sample_name example: '20200514_clip_0_1331_0001'
        #because framerate is 1fps we use framenumber as second
    }

    return targets

def generate_unique_image_id(dateobj,clipname):
    return int( #Using Zfill to zero pad every number 
        f'{dateobj.year}'.zfill(4)+
        f'{dateobj.month}'.zfill(2)+
        f'{clipname.split("_")[1]}'.zfill(3)+
        f'{dateobj.day}'.zfill(2)+
        f'{dateobj.hour}'.zfill(2)+
        f'{dateobj.minute}'.zfill(2)+
        f'{dateobj.second}'.zfill(2))

# COCO META DATA REQUIRED FOR ANNOTATIONS
ANNO_INFO = {
                "description": "Harborfront",
                "version": "1.0",
                "year": 2024,
                "contributor": "Milestone Research Programme at Aalborg University, Visual Analysis and Perception Lab at Aalborg University",
                "date_created": "23/01/2024"
            }

ANNO_LICENSES = [
"COPYRIGHTED IMAGE - PLACEHOLDER LICENSE"
] 

ANNO_CATEGORIES = {
    "human"      : 0,
    "bicycle"    : 1,
    "motorcycle" : 2,
    "vehicle"    : 3,}

if __name__ == "__main__":
    parser  = argparse.ArgumentParser("Generate COCO annotations from Harborfront txt")
    
    #REQUIRED
    parser.add_argument('root', help="Path to the Harborfront Root Directory")
    
    #OPTIONAL
    parser.add_argument('--output', default="./", 
                        help="Output folder for the produced JSON annotation file")
    parser.add_argument('--img_folder', default="frames/", 
                        help="Relative path from root to image directory")
    parser.add_argument('--ann_folder', default="annotations/", 
                        help="Relative path from root to annotation directory")
    parser.add_argument('--meta_file', default=None, 
                        help="Path to the metadata.csv from the original LTD Dataset, [None] to exclude meta data")
    parser.add_argument('--datasplit_yaml', default=None, 
                        help="Optional Dataset split file to generate several JSON annotation files. [None] will result in a complete JSON annotation file")
    args = parser.parse_args()

    #Load splits
    if args.datasplit_yaml is not None:
        data = load_datasplits(args.datasplit_yaml)
    else:
        tmp = {}
        for date in os.listdir(os.path.join(args.root, args.ann_folder)):
            tmp[date] = []
            for clip in os.listdir(os.path.join(args.root, args.ann_folder, date)):
                tmp[date].append(clip) 
        data = {"Annotations" : tmp}

    #Load meta Data
    if args.meta_file is not None:
        meta_df = pd.read_csv(os.path.join(args.root,args.meta_file))

        #Convert datetime string to datetime object
        meta_df["DateTime"] = meta_df["DateTime"].apply(pd.to_datetime)

    #Frame Counter
    frame_idx = 1
    annot_idx = 1

    #Iterate over annotations
    for split in data.keys():
        print(f'Processing Datasplit: {split}')
        samples = []
        #Iterate over data
        for date in tqdm(data[split].keys(), "Processsing Date"):
            for clip in data[split][date]:
                if args.meta_file is not None:

                    #Get Meta Data
                    img_meta_data = meta_df.loc[(meta_df['Clip Name'] == clip) & (meta_df['Folder name'] == int(date))].to_dict("records")[0]

                    #Remove unwanted meta entries
                    for ex in ['Folder name', 'Clip Name', 'DateTime']:
                        del img_meta_data[ex]
                
                #Iterate over frames in current clip
                for frame in os.listdir(os.path.join(args.root, args.img_folder, date, clip)):
                    
                    #Get frame number
                    frame_number = frame.rsplit('.', 1)[0].rsplit('_')[1]
                    
                    #Make file_paths
                    img_path = os.path.join(args.root, args.img_folder, date, clip, "image_{}.jpg".format(frame_number))
                    ann_path = os.path.join(args.root, args.ann_folder, date, clip, "annotations_{}.txt".format(frame_number))
                    
                    #Load annotations
                    sample_name = "{}_{}_{}".format(date, clip, frame_number)
                    annots = load_annotations(sample_name, img_path, ann_path)

                    #Create image and annotation entries
                    img_entry = {
                        #"license": 0,
                        "file_name": os.path.join(args.img_folder, date, clip, "image_{}.jpg".format(frame_number)),
                        #"coco_url": "N/A",
                        #"height": 288,
                        #"width": 384,
                        "date_captured": annots["timestamp"].isoformat(),
                        #"flickr_url": "N/A",
                        #"id": frame_idx,
                        "id": generate_unique_image_id(annots["timestamp"], clip)
                    }
                    
                    #Append metadata
                    if args.meta_file is not None:
                        img_entry["meta"] = img_meta_data

                    #Add object counters
                    for key in ANNO_CATEGORIES.keys():
                        img_entry[key] = 0
                        img_entry[f"{key}_centers"] = []
                    #img_entry["background"] = 0

                    #Process annotations
                    if len(annots["labels"]) >= 1:
                        for i in range(len(annots["labels"])):
                            #Count objects
                            img_entry[annots["labels"][i]] += 1
                            #List Centers
                            img_entry[f'{annots["labels"][i]}_centers'].append(annots["centers"][i])
                    
                    #Append to dataset dict
                    samples.append(img_entry)

        #Turn into pandas dataframe
        dt = pd.DataFrame(samples)
        dt.to_csv(f'{args.output}/{split}_data.csv',";")