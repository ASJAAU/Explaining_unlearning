import itertools
import pandas as pd
import numpy as np
import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset


class REPAIHarborfrontDataset(Dataset):
    CLASS_LIST = {
        0: "human",
        1: "bicycle",
        2: "motorcycle",
        3: "vehicle"
    }

    __KNOWN_TARGETS__ = ['multilabel_counts', 'multilabel_binary', 'counts', 'binary']

    def __init__(self, data_split, root, transform=None, target_transform=None, classes=CLASS_LIST.values(), target_format='multilabel_counts', device='cpu', verbose=False) -> None:
        if verbose:
            print(f'Loading "{data_split}"')
            print(f'Target Classes {classes}')

        #Transform objects
        self.transform = transform
        self.target_transform = target_transform

        #Use target device for storage
        self.device = device

        #Load dataset file
        self.root = root
        data = pd.read_csv(data_split, sep=";")

        #Isolate desired classes
        for c in classes:
            assert c in self.CLASS_LIST.values(), f'{c} is not a known class. \n Known classes:{",".join(self.CLASS_LIST.values())}' 
        self.classes = list(classes)
        
        #Asset target format is known
        assert target_format in self.__KNOWN_TARGETS__, f'{c} is not a known target format. \n Known formats:{", ".join(self.__KNOWN_TARGETS__)}'

        #Create dataset of relevant info
        dataset = {"file_name": list(data['file_name'])}
        for cls in self.classes:
            dataset[f'{cls}'] = data[f'{cls}']

        #Reconstruct Dataframe with only training data
        self.dataset = pd.DataFrame(dataset)

        #Join paths with root
        self.images = self.dataset.apply(lambda x: os.path.join(root, x["file_name"]), axis=1)
        
        #Format labels
        if target_format == 'multilabel_binary': 
            self.labels = self.dataset.apply(lambda x: np.asarray([1 if int(x[g]) > 0 else 0 for g in self.classes],dtype=np.int8), axis=1)
        elif target_format == 'multilabel_counts': 
            self.labels = self.dataset.apply(lambda x: np.asarray([float(x[g]) for g in self.classes],dtype=np.int8), axis=1)

        #Just a sanity check
        if verbose:
            print(f'Successfully loaded "{data_split}" as {self.__repr__()}')
            print("")

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        image = read_image(self.images.iloc[idx])
        label = torch.Tensor(self.labels.iloc[idx])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image.to(self.device), label.to(self.device)
    
    def __repr__(self):
        return self.dataset.__str__() 

    def __str__(self):
        sample=self.__getitem__(0)
        return f'Harborfront Dataset (Pytorch)' + f"\nExample input: {sample[0].shape} \n{sample[0]}" + f"\nExample label: {sample[1].shape} \n{sample[1]}"

    
if __name__ == "__main__":
    import numpy as np
    dataset = REPAIHarborfrontDataset("data/Test_data.csv", "/Data/Harborfront_raw/", verbose=True)
    print(dataset)
    print("------ Dataset overview ------")
    print(dataset.__repr__())
