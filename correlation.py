import seaborn as sns
import pandas as pd
import torch
csv_path = "EURUSD_DATASET/Raw-EURUSD_1.02.2007_26.02.2022.csv"

# read csv
prices = pd.read_csv(csv_path, encoding="UTF8")
dataset = torch.tensor(prices.values[:,1:].tolist())
print(dataset.shape)
corr_dataset = dataset[0:10,1:2]
corr = corr_dataset.corrcoef()

sns.regplot(corr)
print("the end.")