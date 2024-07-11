from matplotlib.pyplot import subplots, show, figure
import numpy as np

def visualize_prediction(image, predictions=None, groundtruth=None, heatmaps=None, classes=None):
    #Heatmaps?
    len_heatmaps = len(heatmaps) if heatmaps is not None else 0
    assert len_heatmaps <= len(classes), "Number of heatmaps may not exceed number of class names"
    
    #Make figure
    fig, axs = subplots(1+len_heatmaps, 1, figsize=(6,6 * (len_heatmaps+1)))

    #Remove graph ticks
    if len_heatmaps < 1:
        #Remove image ticks
        axs.axis('off')

        #Original image
        axs.imshow(image)
        axs.set_title("Input Image")

        if predictions is not None:
            #Check if class names exist
            if classes is None:
                classes = [f'{i}' for i in range(len(predictions))]
            else:
                assert len(predictions) <= len(classes), "The length of 'classes' argument needs to be equal or larger than length of predictions"

            #Write prediction
            text = [f"{classes[i]} - pred:{round(float(predictions[i]),2)} gt: {groundtruth[i] if groundtruth is not None else ''}\n" for i in range(len(predictions))]
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            axs.text(0.05, 0.95, "".join(text), transform=axs.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    else:
        #Remove image ticks
        axs[0].axis('off')

        #Original image
        axs[0].imshow(image)
        axs[0].set_title("Input Image")

    #Visualize heatmaps
    print(f"LENGTH : {len_heatmaps}")
    if len_heatmaps >= 1:
        for i in range(len_heatmaps):
            #Remove graph ticks
            axs[i+1].axis('off')
            #Set image
            axs[i+1].set_title(f"Heatmap: {classes[i]}")
            axs[i+1].imshow(image, aspect='equal')
            #Set heatmap
            hmap = axs[i+1].imshow(np.squeeze(heatmaps[i]), cmap='jet', alpha=0.5)
            #Plot Colorbar / colormap
            cax = axs[i+1].inset_axes([0.2, -0.04, 0.6, 0.04], transform=axs[i+1].transAxes)
            cax.axis('off')
            #fig.colorbar(hmap, cax=cax, orientation='horizontal')

    #Compress padding
    fig.tight_layout()
    
    #Return figure for possible saving
    return fig

