import wandb
import numpy as np

def conf_matrix(preds, labels, idx=None):
    #Stora all matrices
    conf_matrices = {}
    #Set Maximum value
    max_vals = int(labels.max()) + 1
    #Define function to truncate predictions
    set_max = np.vectorize(lambda t: max_vals if t > max_vals else max(t,0))
    #how many matrices we making?
    cls_count = labels.shape[1]
    #Set grid for storage
    grid = np.zeros((cls_count+1, max_vals, max_vals), dtype=int)
    #Iterate over class labels
    for cls in range(cls_count):
        samples = zip(set_max(preds[:,cls]), labels[:,cls])
        for point in samples:
            grid[0,int(point[0]),int(point[1])] += 1
            grid[cls+1,int(point[0]),int(point[1])] += 1
    #name conf_matrices
    conf_matrices["conf_matrix"]= grid[0,:,:]
    for i in range(cls_count-1):
        conf_matrices[f"conf_matrix_cls{i}"] = grid[i+1,:,:]
    return conf_matrices


if __name__ == '__main__':
    gt = np.random.rand(10000,1)*16
    gt = gt.astype(np.int8)
    noise = (np.random.rand(gt.shape[0], gt.shape[1])-0.5)*10
    print(conf_matrix(gt,gt))
    print(conf_matrix(gt,gt+noise))