import warnings
from torch.utils.data import Dataset
import torch
import pandas as pd
import torchvision.transforms as trans

import matplotlib.pylab as plt

plt.ion()  # interactive mode

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Forex_Dataset(Dataset):
    """ Reading csv files downloaded from forext sites for prediction."""
    def __init__(self, csv_file, transform=None, seq_lenght = 5) -> None:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()

        self.seq_len = seq_lenght
        self.transform = transform
        
        # read CSV dataset and change it to torch.tensor type
        self.dataset_csv = pd.read_csv(csv_file, encoding="UTF=8")
        self.dataset = torch.tensor(self.dataset_csv.values[:,1:].tolist())
        
        # normalize dataset based on its features.
        for i in range(self.dataset.shape[1] -1):
            self.dataset[:,i] = (self.dataset[:,i] - self.dataset[:,i].mean()) / self.dataset[:,i].std()
        # get headers of dataset without timestamp colums,the first columns.
        self.headers = self.dataset_csv.columns.tolist()[1:]


    def __len__(self):
        """
            return the length of the dataset
        """
        return(len(self.dataset_csv))

    
    def __getitem__(self, index):
        """
            return samples based on index and sequence length.
            `return type`: torch.tensor
        """

        samples = self.dataset[index : index + self.seq_len,:-1]
        # try to find the `close price` of the next day 
        target = self.dataset[index + self.seq_len - 1, -1]
        
        # return super().__getitem__(index)
        # print(f"samples:\n {samples}")
        # print(f"target:\n {target}")
        return samples, target
    
    def header(self):
        """
            return the headers title of the dataset
        """
        return self.headers
    

# there is no train of test dataset split
