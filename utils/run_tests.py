#!/usr/bin/env python
import sys
import subprocess
import unittest

#Dataset testing
from utils.dataloader import REPAIHarborfrontDataset
from torch.utils.data import DataLoader

#Metrics testing
import utils.metrics as metrics
from functools import partial

#Import for logging text
from utils.utils import get_config, Logger
import numpy as np


__CONFIG__ = "./configs/mu/config_mu_fine.yaml"

class Test_Dataloader(unittest.TestCase):
    
    def test_dataset(self):
        dataset = REPAIHarborfrontDataset("data/Test_data.csv", "data/example_data/", verbose=True)
        print(dataset)
        print("------ Dataset overview ------")
        print(dataset.__repr__())
        return True
    
    def test_dataloader(self):
        dataset = REPAIHarborfrontDataset("data/Test_data.csv", "data/example_data/")
        dataloader = DataLoader(
        dataset, 
        batch_size=12, 
        shuffle=True
        )
        #print example batch (sanity check)
        dummy_sample = next(iter(dataloader))
        print(f"Input Tensor = {dummy_sample[0].shape}")
        print(f"Label Tensor = {dummy_sample[1].shape}")
        return True
    
class Test_Metrics(unittest.TestCase):
    def test_metrics(self):
        #Create dummy testing samples
        gt = np.asarray([
            [0,1,8,0],
            [12,1,0,0],
            [1,14,0,0],
            [1,1,2,1],],dtype=np.float32)
        print("Testing metrics with no errors",     [f(gt,gt)                                 for f in metrics.__all__()])
        print("Testing metrics with small errors",  [f(gt+(np.random.rand(gt.shape[0], gt.shape[1])*0.2), gt) for f in metrics.__all__()])
        print("Testing metrics with large errors",  [f(gt+(np.random.rand(gt.shape[0], gt.shape[1])*10),  gt) for f in metrics.__all__()])
        for i in range(gt.shape[1]):
            print(f"classwise ({i}) no error",     [f(gt,gt, idx=i)for f in metrics.__all__()])
            print(f"classwise ({i}) mid error",    [f(gt+(np.random.rand(gt.shape[0], gt.shape[1])*0.2), gt, idx=i) for f in metrics.__all__()])
            print(f"classwise ({i}) high error",   [f(gt+(np.random.rand(gt.shape[0], gt.shape[1])*10),  gt, idx=i) for f in metrics.__all__()])

class Test_Logger(unittest.TestCase):
    def test_LogOutput(self):
        print("Testing ")
        #Retrieve Config
        config = get_config(__CONFIG__)
        #Disable logging to WANDB
        config["wandb"]["enabled"] = False
        print(config)
        #Create logger
        MetricLogger = Logger(config, "./", classwise_metrics=config["data"]["classes"])
        #Create dummy samples
        pred= np.asarray([
            [1,1,0,1],
            [1,1,1,0],
            [1,0,0,0],
            [0,1,2,1],],dtype=np.float32)
        gt = np.asarray([
            [0,1,0,0],
            [1,1,0,0],
            [1,1,0,0],
            [1,1,2,1],],dtype=np.float32)
        #Test with no error
        MetricLogger.add_prediction(gt, gt)
        print(MetricLogger.log(prepend="NE_"))
        #Test with error
        MetricLogger.add_prediction(pred, gt)
        print(MetricLogger.log(prepend="WE_"))
        return True

    def test_Log(self):
        return False

if __name__ == '__main__':
    unittest.main()