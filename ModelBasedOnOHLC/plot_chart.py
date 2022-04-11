import pandas as pd
from datetime import datetime
import mplfinance as mpl
from parameters import dataset_filePath

# start_date="2020-01-01"
# end_date="2020-02-01"
# dates = pd.date_range(start=start_date, end=end_date)
# print(dates)

# day1  = ["1.30140","1.30615","1.29474","1.29580"]
# day2 = ["1.29580","1.29580","1.29580","1.29580"]
# day3 = ["1.29532","1.29646","1.29498","1.29582"]

# daily = pd.read_csv('data_chart.csv',index_col=0,parse_dates=True)
daily = pd.read_csv('chart.csv', index_col=0, parse_dates=True)
daily.index.name = 'Date'
# print(daily.shape)
# print(daily.head(3))
# print(daily.tail(3))

mpl.plot(daily,type='candle', show_nontrading=True)

print("The end.")