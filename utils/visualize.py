from matplotlib.pyplot import subplots, show, figure
import numpy as np

def visualize_heatmap(image, heatmap, cmap='jet'):
    fig, ax = subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('tight')
    ax.axis('off')

    #Original image
    ax.imshow(heatmap, cmap=cmap, alpha=1.0)
    ax.axis('off')
    return fig

def visualize_prediction(image, predictions=None, groundtruth=None, heatmaps=None, classes=None, cmap='jet'):
    #Heatmaps?
    len_heatmaps = len(heatmaps) if heatmaps is not None else 0
    
    #Make figure
    fig, axs = subplots(1+len_heatmaps, 1, figsize=(6,6 * (len_heatmaps+1)))

    #Remove graph ticks
    if len_heatmaps < 1:
        #Remove image ticks
        axs.axis('off')

        #Original image
        axs.imshow(image, cmap='gray', vmin=0, vmax=255)
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
        axs[0].imshow(image, cmap='gray', vmin=0, vmax=255)
        axs[0].set_title("Input Image")

        #Define classes
        if classes is None:
            classes = [f'{i}' for i in range(len(predictions))]
        else:
            assert len(len_heatmaps) <= len(classes), "The length of 'heatmaps' argument needs to be equal or larger than length of predictions"

        #Write prediction
        text = [f"{classes[i]} - pred:{round(float(predictions[i]),2)} gt: {groundtruth[i] if groundtruth is not None else ''}\n" for i in range(len(predictions))]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axs[0].text(0.05, 0.95, "".join(text), transform=axs[0].transAxes, fontsize=12, verticalalignment='top', bbox=props)


        #Draw heatmaps
        for i in range(len_heatmaps):
            #Remove graph ticks
            axs[i+1].axis('off')
            #Set image
            axs[i+1].set_title(f"Heatmap: {classes[i]}")
            axs[i+1].imshow(image, aspect='equal', cmap='gray')
            #Set heatmap
            hmap = axs[i+1].imshow(heatmaps[i], cmap=cmap, alpha=0.5)
            #Plot Colorbar / colormap
            cax = axs[i+1].inset_axes([0.2, -0.04, 0.6, 0.04], transform=axs[i+1].transAxes)
            cax.axis('off')
            fig.colorbar(hmap, cax=cax, orientation='horizontal')
            

    #Compress padding
    fig.tight_layout()
    
    #Return figure for possible saving
    return fig

