import pandas as pd
from datetime import datetime
import mplfinance as mpl
from parameters import dataset_filePath


def plot_chart_price():
        
    # daily = pd.read_csv('data_chart.csv',index_col=0,parse_dates=True)
    daily = pd.read_csv('chart.csv', index_col=0, parse_dates=True)
    daily.index.name = 'Date'
    # print(daily.shape)
    # print(daily.head(3))
    # print(daily.tail(3))

    mpl.plot(daily,type='candle', show_nontrading=True)
