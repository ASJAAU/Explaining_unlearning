import numpy as np
from functools import partial

def get_metrics(target_format, classwise=[]):
    if 'count' in target_format:
        metrics = {}
        metrics["MAE"] = mae
        #metrics["MAPE"] = mape #Yields NaN when gt is 0
        metrics["RMSE"] = rmse
        #metrics["R2"] = r2 
        
        #Classwise metrics
        for i,c in enumerate(classwise):
            metrics[f"MAE_{c}"] = partial(mae, idx=i)
            #metrics[f"MAPE_{c}"] = partial(mape, idx=i) #Yields NaN when gt is 0
            metrics[f"RMSE_{c}"] = partial(rmse, idx=i)
            #metrics[f"R2_{c}"] = partial(r2, idx=i) #Not implemented
        return metrics
    else:
        print("DONT RECOGNIZE THE TARGET FORMAT, OMITTING EVALUATION METRICS")
        return {}

class Logger:
    def __init__(self, cfg, out_folder, classwise_metrics=[]) -> None:
        #Save local copies of relevant variables
        self.metrics = get_metrics(cfg["data"]["target_format"], classwise_metrics)
        self.output_path = out_folder
        self.classes = cfg["data"]["classes"]

        #Init wandb
        if cfg["wandb"]["enabled"]:
            self.wandb = wandb_logger(
                cfg,
                output_path=out_folder,
                )
        else:
            self.wandb = None

        #Buffer for predictions
        self.preds = []
        self.labels= []

    def clear_buffer(self):
        self.preds = []
        self.labels= []

    def log(self, xargs={}, clear_buffer=True, prepend=''):
        preds = np.concatenate(self.preds, axis=0)
        labels= np.concatenate(self.labels, axis=0)

        # Create step log
        to_log = {}

        #Add xargs to log
        for k,v in xargs:
            to_log[k]=v

        # Apply metrics
        for name, fn in self.metrics.items():
            to_log[f'{prepend}_{name}'] = fn(preds, labels)

        #upload to wandb
        if self.wandb is not None:
            self.wandb.log(to_log)

        #Clear buffer post logging
        if clear_buffer:
            self.clear_buffer()

        #return results
        return to_log
    
    def add_prediction(self, prediction, label):
        self.preds.append(prediction)
        self.labels.append(label)

def wandb_logger(cfg, output_path="./wandb"):
    import wandb
    if cfg["wandb"]["enabled"]:
        wandb_logger = wandb.init(
            project=cfg["wand"]["project_name"],
            config=cfg,
            tags=cfg["wandb"]["tags"],
            dir=output_path,
        )
        return wandb_logger
    else:
        return None
    
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

def rmse(preds, labels, idx=None):
    if idx is None: #Calculating total RMSE
        return np.sqrt(np.mean(abs(preds-labels)**2))
    else: #Calculating class specific RMSE
        return np.sqrt(np.mean(abs(preds[:,idx]-labels[:,idx])**2))

def r2(preds, labels, idx=None):
    return None

if __name__ == '__main__':
    import yaml
    from saving import get_config
    config = get_config("./configs/config.yaml")
    config["wandb"]["enabled"] = False
    MetricLogger = Logger(config, "./", classwise_metrics=config["data"]["classes"])

    pred = np.asarray([
        [1,1,0,0],
        [1,1,0,0],
        [1,1,0,0],
        [1,1,2,0],
        ],
        dtype=np.float32)
    
    gt = np.asarray([
        [0,1,0,0],
        [1,1,0,0],
        [1,1,0,0],
        [1,1,2,0],
        ],
        dtype=np.float32)
    
    # Add multiple batches just for sanity check
    MetricLogger.add_prediction(pred, gt)
    MetricLogger.add_prediction(pred, gt)

    print(MetricLogger.log(prepend="test"))

