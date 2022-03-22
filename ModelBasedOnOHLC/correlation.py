import seaborn as sns
import pandas as pd
import torch

csv_path = "ModelBasedOnOHLC/EURUSD_DATASET/EURUSD_No_Volume_Buy.csv"

# read csv
prices = pd.read_csv(csv_path, encoding="UTF8")
dataset = torch.tensor(prices.values[:,1:].tolist())
print(dataset.shape)
corr_dataset = dataset[0:10,1:2]
corr = corr_dataset.corrcoef()

sns.regplot(corr)
print("the end.")