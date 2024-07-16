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
                #Add bounding box
                boxes.append([int(row[2]), int(row[3]), int(row[4]), int(row[5])]) #Xtopleft, Ytopleft, Xbottomright, Ybottomright
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

ANNO_CATEGORIES = {
    "human"      : 0,
    "bicycle"    : 1,
    "motorcycle" : 2,
    "vehicle"    : 3,}

if __name__ == "__main__":
    parser  = argparse.ArgumentParser("Generate MY/XAI CSV from Harborfront txt")
    
    #REQUIRED
    parser.add_argument('root', help="Path to the Harborfront Root Directory")
    
    #OPTIONAL
    parser.add_argument('--output', default="./", help="Output folder for the produced JSON annotation file")
    parser.add_argument('--img_folder', default="frames/", help="Relative path from root to image directory")
    parser.add_argument('--ann_folder', default="annotations/", help="Relative path from root to annotation directory")
    args = parser.parse_args()

    #Load Data
    tmp = {}
    for date in os.listdir(os.path.join(args.root, args.ann_folder)):
        tmp[date] = []
        for clip in os.listdir(os.path.join(args.root, args.ann_folder, date)):
            tmp[date].append(clip) 
    data = {"Annotations" : tmp}

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
                        "file_name": os.path.join(args.img_folder, date, clip, "image_{}.jpg".format(frame_number)),
                        "date_captured": annots["timestamp"].isoformat(),
                        "id": generate_unique_image_id(annots["timestamp"], clip)
                    }

                    #Add object counters
                    for key in ANNO_CATEGORIES.keys():
                        img_entry[key] = 0

                    #Process annotations
                    if len(annots["labels"]) >= 1:
                        for i in range(len(annots["labels"])):
                            #Count objects
                            img_entry[annots["labels"][i]] += 1
                    
                    #Append to dataset dict
                    samples.append(img_entry)

        #Turn into pandas dataframe
        dt = pd.DataFrame(samples)
        dt.to_csv(f'{args.output}/{split}_data.csv',";")