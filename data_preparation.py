import pandas as pd
import torch
import numpy
from torch.utils.data import Subset
from torch.utils import data


class My_custom_dataset():
    def __init__(self, csv_file: str = r"EURUSD_DATASET/EURUSD_1.02.2007_26.02.2022.csv", 
                return_type=torch.tensor, test_split=0.25) -> None:
        self.csv_file = csv_file,
        self.return_type = return_type
        self.test_split = test_split
        self.dataset = []


        # loading csv file and put its data to 'data' variable
        self.raw_data = pd.read_csv(r"EURUSD_DATASET/EURUSD_1.02.2007_26.02.2022.csv",
                            encoding = "UTF-8")

        # removing Gmt time from headers. Thus keep just open, high, low, close, volume and buy items.
        self.headers = self.raw_data.columns.to_list()[1:]
        


    # return specific data from datastore
    def dataset_Fetch(self, csv_data: pd=data, columns: str=["*"]):
        r'''
        parameters:
        ----------
        `csv_data`: pandas csv file path as read. `Default: data`

        `columns`: str, It shoule be fit into brackets, i.e. ["Open"] or ["Open", "Close"]

        columns headers: "Gmt time", "Open", "High", "Low", "Close", "Volume", "Buy" 
                        or any combination of these headers.

        `retur_type`: Type of returning values for further manipulation.
                        It could be torch.tensor or numpy
                        Default: torch.tensor
        
        return:
        -------
        Data as torch tensor or numpy for manipulation. additionally headers.
        All of these returned as a tuple.
        '''
        
        
        if columns == ["*"]:
            # self.dataset = csv_data
            self.dataset = self.raw_data
        else:
            df = pd.DataFrame(data, columns= columns)
            self.dataset = df
        
        self.dataset = self.dataset.to_numpy()

        if self.return_type == numpy:
            self.dataset = numpy.float32(self.dataset[:,1:])
            return self.dataset, self.headers
        elif self.return_type == torch.tensor:
            self.dataset = torch.tensor(numpy.float32(self.dataset[:,1:]))
            return self.dataset, self.headers
        else:
            raise TypeError("Only torch.tensor or numpy is acceptable for return type.")
            

    def train_test_dataset_split(self, test_split=0.25):
        trainset = data.splli

    # print("End of data preparation section")