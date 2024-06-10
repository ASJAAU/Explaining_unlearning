import wandb
import numpy as np
import matplotlib.pyplot as plt

def conf_matrix(preds, labels, idx=None):
    #Set Maximum value
    max_vals = int(labels.max()) + 1
    #Define function to truncate predictions
    set_bounderies = np.vectorize(lambda t: max_vals if t > max_vals else int(max(t,0)))
    #how many matrices we making?
    cls_count = labels.shape[1]
    #Set grid for storage
    preds=set_bounderies(preds)
    # #Make 2D Histogram
    if idx is None:
        xs, ys = [], []
        for cls in range(cls_count):
            xs.extend(set_bounderies(preds[:,cls]))
            ys.extend(labels[:,cls])
        return wandb.plot.confusion_matrix(y_true=ys, preds=xs, class_names=np.linspace(0,max_vals, max_vals+1).astype(np.uint8))
    else:
        wandb.plot.confusion_matrix(y_true=labels[:,idx], preds=set_bounderies(preds[:,idx]), class_names=np.linspace(0,max_vals, max_vals+1).astype(np.uint8))

def conf_matrix_plot(preds, labels, idx=None):
    #Set Maximum value
    max_vals = int(labels.max()) + 1
    #Define function to truncate predictions
    set_bounderies = np.vectorize(lambda t: max_vals if t > max_vals else max(t,0))
    #how many matrices we making?
    cls_count = labels.shape[1]
    #Set grid for storage
    preds=set_bounderies(preds)
    #Make 2D Histogram
    if idx is None:
        conf_matrices = np.zeros((max_vals, max_vals))
        for cls in range(cls_count):
            temp, xedges, yedges = np.histogram2d(x=preds[:,cls],y=labels[:,cls],bins=(np.linspace(0,max_vals, max_vals+1).astype(np.uint8), np.linspace(0,max_vals, max_vals+1).astype(np.uint8)))
            conf_matrices += temp
        #Normalize with respect to total of a specific class
        for i in range(conf_matrices.shape[0]):
            for j in range(conf_matrices.shape[1]):
                conf_matrices[i,j] = conf_matrices[i,j]/(np.sum(conf_matrices[i,:]+np.sum(conf_matrices[:,j]-conf_matrices[i,j]))) if (np.sum(conf_matrices[i,:]+np.sum(conf_matrices[:,j]-conf_matrices[i,j]))) > 0 else conf_matrices[i,j]
        return wandb.Image(plt.imshow(conf_matrices.T, cmap="jet", origin="lower"))
    else:
        conf_matrices, xedges, yedges = np.histogram2d(x=preds[:,idx],y=labels[:,idx],bins=np.linspace(0,max_vals,max_vals).astype(np.uint8))
        #Normalize with respect to total of a specific class
        for i in range(conf_matrices.shape[0]):
            for j in range(conf_matrices.shape[1]):
                conf_matrices[i,j] = conf_matrices[i,j]/(np.sum(conf_matrices[i,:]+np.sum(conf_matrices[:,j]-conf_matrices[i,j]))) if (np.sum(conf_matrices[i,:]+np.sum(conf_matrices[:,j]-conf_matrices[i,j]))) > 0 else conf_matrices[i,j]
        return wandb.Image(plt.imshow(conf_matrices.T, cmap="jet", origin="lower"))

if __name__ == '__main__':
    gt = np.random.rand(10000,1)*16
    gt = gt.astype(np.int8)
    noise = (np.random.rand(gt.shape[0], gt.shape[1])-0.5)*10
    print(conf_matrix(gt,gt))
    print(conf_matrix(gt,gt+noise))