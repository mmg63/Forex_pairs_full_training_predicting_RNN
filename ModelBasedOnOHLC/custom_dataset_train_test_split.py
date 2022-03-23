import warnings
from torch.utils.data import Dataset
import torch
import pandas as pd

import matplotlib.pylab as plt

plt.ion()  # interactive mode

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Forex_train_Dataset(Dataset):
    """ Reading csv files downloaded from forext sites for prediction."""
    def __init__(self, csv_file, transform=None, seq_lenght = 5, train_size_ratio=0.8) -> None:
        """
        Args:
            `csv_file` (string): Path to the csv file with annotations.
            `root_dir` (string): Directory with all the images.
            `transform` (callable, optional): Optional transform to be applied
                on a sample.
            `train_size_ratio` (float) Amount of dataset used for train-set. Default 0.8
            
        """
        super().__init__()

        self.seq_len = seq_lenght
        self.transform = transform
        self.train_size_ratio = 0.8
        
        # read CSV dataset and change it to torch.tensor type
        self.dataset_csv = pd.read_csv(csv_file, encoding="UTF=8")
        # remove date from dataset in order to prepare data for test
        self.dataset = torch.tensor(self.dataset_csv.values[:,1:].tolist())
        # get train samples from the dataset and set it again to the dataset
        len_dataset = len(self.dataset)
        
        self.dataset = self.dataset[:int(len_dataset*self.train_size_ratio), :]
        
        # get dates for plotting candles if it is neccessary 
        self.dates = self.dataset_csv.values[:int(len_dataset*self.train_size_ratio),0].tolist()
        


        # normalize dataset based on its features.
        for i in range(self.dataset.shape[1] -4):   # -4 is for just normalizing input features
            # self.dataset[:,i] = (self.dataset[:,i] - self.dataset[:,i].mean()) / self.dataset[:,i].std()
            # Normalize features from 0 to 1
            self.dataset[:, i] -= self.dataset[:, i].min()
            self.dataset[:, i] /= self.dataset[:, i].max()

        # get headers of dataset without timestamp colums,the first columns.
        self.headers = self.dataset_csv.columns.tolist()[1:]


    def get_dates(self):
            return self.dates


    def __len__(self):
        """
            return the length of the dataset
        """
        return(len(self.dataset))

    
    def __getitem__(self, index):
        """
            return samples based on index and sequence length.
            `return type`: torch.tensor
        """
        # print(f"samples: {self.dataset[index + self.seq_len - 1,:]}")

        samples = self.dataset[index : index + self.seq_len,:-4]
        # try to find the `close price` of the next day 
        # target_Open = self.dataset[index + self.seq_len - 1, -4]
        # target_High = self.dataset[index + self.seq_len - 1, -3]
        # target_Low = self.dataset[index + self.seq_len - 1, -2]
        target_Close = self.dataset[index + self.seq_len - 1, -1]
        
        # return super().__getitem__(index)
        # print(f"samples:\n {samples}")
        # print(f"target:\n {target}")
        return samples, target_Close  #target_Open, target_High, target_Low, target_Close 
    
    def header(self):
        """
            return the headers title of the dataset
        """
        return self.headers
    

class Forex_test_Dataset(Dataset):
    """ Reading csv files downloaded from forext sites for prediction."""
    def __init__(self, csv_file, transform=None, seq_lenght = 5, train_size_ratio=0.2) -> None:
        """
        Args:
            `csv_file` (string): Path to the csv file with annotations.
            `root_dir` (string): Directory with all the images.
            `transform` (callable, optional): Optional transform to be applied
                on a sample.
            `train_size_ratio` (float) Amount of dataset used for train-set. Default 0.8
            
        """
        super().__init__()

        self.seq_len = seq_lenght
        self.transform = transform
        self.train_size_ratio = 0.8
        
        # read CSV dataset and change it to torch.tensor type
        self.dataset_csv = pd.read_csv(csv_file, encoding="UTF=8")
        # remove date from dataset in order to prepare data for test
        self.dataset = torch.tensor(self.dataset_csv.values[:,1:].tolist())
        
        # get train samples from the dataset and set it again to the dataset
        len_dataset = len(self.dataset)
        # print(len_dataset)
        # self.dataset = self.dataset[int(len_dataset * self.train_size_ratio):, :]
        # print(len(self.dataset))

        # get dates for plotting candles if it is neccessary 
        self.dates = self.dataset_csv.values[int(len_dataset * self.train_size_ratio):, 0].tolist()

        # normalize dataset based on its features.
        for i in range(self.dataset.shape[1] -4):   # -4 is for just normalizing input features
            # self.dataset[:,i] = (self.dataset[:,i] - self.dataset[:,i].mean()) / self.dataset[:,i].std()
            # Normalize features from 0 to 1
            self.dataset[:, i] -= self.dataset[:, i].min()
            self.dataset[:, i] /= self.dataset[:, i].max()

        # get headers of dataset without timestamp colums,the first columns.
        self.headers = self.dataset_csv.columns.tolist()[1:]


    def __len__(self):
        """
            return the length of the dataset
        """
        return(len(self.dataset))


    def get_dates(self):
            return self.dates


    def __getitem__(self, index):
        """
            return samples based on index and sequence length.
            `return type`: torch.tensor
        """

        samples = self.dataset[index : index + self.seq_len,:-4]
        # try to find the `close price` of the next day 
        # target_Open = self.dataset[index + self.seq_len - 1, -4]
        # target_High = self.dataset[index + self.seq_len - 1, -3]
        # target_Low = self.dataset[index + self.seq_len - 1, -2]
        target_Close = self.dataset[index + self.seq_len - 1, -1]
        
        # return super().__getitem__(index)
        # print(f"samples:\n {samples}")
        # print(f"target:\n {target}")
        return samples,target_Close  # target_Open, target_High, target_Low, target_Close 
    
    def header(self):
        """
            return the headers title of the dataset
        """
        return self.headers
    

# there is no train of test dataset split

# there is no train of test dataset split
