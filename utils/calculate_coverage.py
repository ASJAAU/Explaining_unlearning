import numpy as np
import argparse
import os
import glob
from utils.metrics import HmCvr
from tqdm import tqdm
from utils.utils import existsfolder
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

if __name__ == "__main__":
    classes = ["human","bicycle","motorcycle","vehicle"]
    #CLI
    parser = argparse.ArgumentParser("Calculate coverage and attentionshift")
    #Positionals
    parser.add_argument("inputs", type=str, nargs= '+', help="Path to files/folders to run inference on")
    parser.add_argument("--classes", type=str, nargs= '+', default=classes,help="classses to calculate coverage from", choices=classes)

    #Optional
    parser.add_argument("--remove_overlap", type=str, nargs= '+', default=None,help="classses to calculate coverage from", choices=classes)
    parser.add_argument("--original", default=None, help="The name of the original model (to use as origin for shift calculation)")
    parser.add_argument("--explanation_root", default="/Data/Harborfront_raw/repai/model_explinations/", help="Path to the folder containing model explinations")
    parser.add_argument("--colormap", default='jet', help="Which Matplotlib Colormap to use")
    args = parser.parse_args()


    #Parse each model
    for model in args.inputs:
        #Retrieve all model explinations
        explanation_paths = args.explanation_root + "/" + model + "/heatmap/"
        heatmap_paths = glob.glob(explanation_paths+"/**/*.npy", recursive=True)
        #Store heatmap coverage
        HmCvr_values = []
        shift_stds = []
        shift_maps = []
        largest_shift = 0
        for hmap_path in heatmap_paths:
            #Load class specified masks
            masks = []
            classes_present = {}
            hmap = np.load(hmap_path)
            hmap /= np.sum(hmap)
            #print(hmap.shape)
            for cls in args.classes:
                mask_path = hmap_path.replace("heatmap", "mask")[:-4] + f"_{cls}.npy"
                mask = np.where(np.load(mask_path)>0, 1, 0)
                masks.append(mask)
                #Check if any of this class are present
                classes_present[cls] = np.sum(mask)>0
            
            #Collapse to single mask
            combined_mask = np.stack(masks, axis=0)
            final_mask = np.where(np.sum(combined_mask, axis=0) > 0, 1, 0)

            #Remove overlap with classes
            if args.remove_overlap is not None:
                for overlap in args.remove_overlap:
                    mask_path = hmap_path.replace("heatmap", "mask")[:-4] + f"_{overlap}.npy"
                    mask = np.where(np.load(mask_path)>0, 1, 0)
                    #tmp = final_mask
                    final_mask = np.where((final_mask-mask) > 0, 1, 0)
                    #print(f"removed {overlap}")
                    #print(f"{abs(np.sum(tmp)-np.sum(final_mask))} values changed")


            #Only calculate for image with objects of interest in them
            if any(classes_present.values()):
                #Calculate HmCvr
                #print(hmap.shape, final_mask.shape)
                score = HmCvr(hmap, final_mask)
                HmCvr_values.append(score)

            if args.original is not None:
                org_hmap = np.load(hmap_path.replace(model, args.original))
                org_hmap /= np.sum(org_hmap)
                #print(hmap.shape, org_hmap.shape)
                shift = hmap - org_hmap
                shift_stds.append(np.std(shift))
                existsfolder(os.path.dirname(hmap_path.replace("heatmap", "shift")))
                #print(shift.shape)
                np.save(hmap_path.replace("heatmap", "shift"), shift)
                shift_maps.append((shift, hmap_path.replace("heatmap", "shift")[:-4] + f".jpg"))
                largest_shift = max(np.abs(shift).max(), largest_shift)
                #att_image = LinearSegmentedColormap.from_list('rg',['r','w','g'], N=256)()
                #plt.imsave(hmap_path.replace("heatmap", "shift")[:-4] + f".jpg", att_image)


        #print(HmCvr_values)
        print(f"{model} - HmCvr:{np.sum(HmCvr_values)/len(HmCvr_values)}")
        if args.original is not None:
            print(f"{model} - AttShift: {np.sum(shift_stds)/len(shift_stds)}")
            cmapping = LinearSegmentedColormap.from_list('rg', ['r','w','g'],N=256)
            largest_shift *= 0.5 #Scaling factor for visualization
            if largest_shift <= 0:
                print(f"WARNING: Shift magnitude is {largest_shift} for {model}")
            for shiftmap in shift_maps:
                plt.imsave(shiftmap[1], shiftmap[0], cmap=cmapping, vmax=largest_shift, vmin=-largest_shift)












                





