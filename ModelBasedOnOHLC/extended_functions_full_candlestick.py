import pandas as pd
import matplotlib.pyplot as plt
from parameters import *
import torch
import numpy as np
import parameters
import mplfinance as mpf
from time import ctime
from datetime import datetime, date
plt.ion()


def plot_acc_loss(data:list, caption):
    # Plot figure
    plt.figure(figsize=(10,5))
    plt.title(f"Training and Validation Loss with lr = {lr}")
    # plt.plot(data, label="val")
    plt.yticks(np.arange(0, 1, step=0.001))
    plt.plot(data, label=caption)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(b=True, color = 'green', linestyle = '--', linewidth = 0.5)
    plt.show()
    plt.pause(20)


def plot_test_values_predicted(real_values:list, 
                                predicted_values:list,  
                                caption_for_real_values, 
                                caption_for_predicted_values,
                                y_label="Close Price"):
    # Plot figure
    plt.figure(figsize=(10,5),)
    plt.title("Real test and predicted values")
    # plt.plot(data, label="val")
    plt.grid()
    plt.plot(real_values, label=caption_for_real_values)
    plt.plot(predicted_values, label=caption_for_predicted_values)
    plt.xlabel("Days")
    plt.ylabel(y_label)
    plt.legend(["Real price","Predicted price"])
    plt.show()


def plot_price(price_list):
    # show close prices in python plot
    plt.ion()
    plt.plot(price_list)
    plt.annotate("close price", [0.,0.])
    plt.show()
    plt.pause(10)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = optimizer.parag_gropus[0]['lr'] * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# saving model parameters
def save_model(model_Open, model_High, model_Low, model_Close, 
                optimizer_Open, optimizer_High, Optimizer_Low, Optimizer_Close,
                epoch, 
                loss_Open, loss_High, loss_Low, loss_Close):
    state_dicts = { 'model_Open_state_dict': model_Open.state_dict(),
                    'model_Hight_state_dict': model_High.state_dict(),
                    'model_Low_state_dict': model_Low.state_dict(),
                    'model_Close_state_dict': model_Close.state_dict(),
                    'optimizer_Open_state_dict': optimizer_Open.state_dict(), 
                    'optimizer_High_state_dict': optimizer_High.state_dict(), 
                    'optimizer_Low_state_dict': Optimizer_Low.state_dict(), 
                    'optimizer_Close_state_dict': Optimizer_Close.state_dict(), 
                    'epoch': epoch, 
                    'loss_Open': loss_Open,
                    'loss_High': loss_High,
                    'loss_Low': loss_Low,
                    'loss_Close': loss_Close
                    }
    model_state_dict_path = parameters.model_state_dict_path
    torch.save(state_dicts, model_state_dict_path)


# saving model parameters
def save_single_model(model_Close, 
                Optimizer_Close,
                epoch, 
                loss_Close):
    state_dicts = {
                    'model_Close_state_dict': model_Close.state_dict(),
                    'optimizer_Close_state_dict': Optimizer_Close.state_dict(), 
                    'epoch': epoch, 
                    'loss_Close': loss_Close
                    }
    model_state_dict_path = parameters.model_state_dict_path
    torch.save(state_dicts, model_state_dict_path)


def plot_price_chart(prices):
    # source_csv = pd.read_csv(reader=model_state_dict_path, encoding="UTF8")
    # candles = pd.read_csv(dataset_filePath,index_col=0,parse_dates=True)
    # dates = candles.values[int(len(candles) * 0.8):, 0].tolist()
    candles = pd.DataFrame(prices).transpose()
    candles.columns = ['Date', 'Open', 'High', 'Low', 'Close']
    candles['Date'] = pd.to_datetime(candles['Date'])
    # candles.index.name = 'Date'
    mpf.plot(candles)
    print("end plot chart")


def convert_list_to_csv(open,high,low,close):
    prices = [open, high, low, close]
    datelist = pd.date_range(datetime.today().date(), periods=100).tolist()
    pass


def date_range(start_date, end_date):
    # start="2020-01-01", end="2020-02-01"
    pd.date_range(start=start_date, end=end_date)
    
