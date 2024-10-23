import numpy as np
from functools import partial

def __all__():
    return [mae,mape,mase,rmse]

def mae(preds, labels, idx=None):
    if idx is None: #Calculating total MAE
        return np.mean(abs(preds-labels))
    else: #Calculating class specific MAE
        return np.mean(abs(preds[:,idx]-labels[:,idx]))
    
def mape(preds, labels, idx=None):
    if idx is None: #Calculating total MAPE
        return np.mean((abs(preds-labels)/labels)*100)
    else: #Calculating class specific MAPE
        return np.mean((abs(preds[:,idx]-labels[:,idx])/labels)*100)
    
def mase(preds, labels, idx=None):
    if idx is None: #Calculating total MASE
        return np.mean(abs(preds-labels)) / np.mean(abs(labels-np.tile(np.mean(labels, axis=0),(labels.shape[0],1))))
    else: #Calculating class specific MASE
        return np.mean(abs(preds[:,idx]-labels[:,idx])) / np.mean(abs(labels[:,idx]-np.tile(np.mean(labels[:,idx], axis=0),(labels.shape[0],1))))

def rmse(preds, labels, idx=None):
    if idx is None: #Calculating total RMSE
        return np.sqrt(np.mean(abs(preds-labels)**2))
    else: #Calculating class specific RMSE
        return np.sqrt(np.mean(abs(preds[:,idx]-labels[:,idx])**2))

def r2(preds, labels, idx=None):
    return None

def HmCvr(heatmap, mask):
    return np.sum(np.multiply(heatmap,mask)) / np.sum(heatmap)
