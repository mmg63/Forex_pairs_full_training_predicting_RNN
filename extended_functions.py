import matplotlib.pyplot as plt
from parameters import *
import torch
import numpy as np
import parameters
plt.ion()


def plot_acc_loss(data:list, caption):
    # Plot figure
    plt.figure(figsize=(10,5))
    plt.title(f"Training and Validation Loss with lr = {lr}")
    # plt.plot(data, label="val")
    plt.yticks(np.arange(0, 2, step=0.001))
    plt.plot(data, label=caption)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(b=True, color = 'green', linestyle = '--', linewidth = 0.5)
    plt.show()
    plt.pause(20)


def plot_test_values_predicted(real_values:list,predicted_values:list,  caption_for_real_values, caption_for_predicted_values):
    # Plot figure
    plt.figure(figsize=(10,5))
    plt.title("Real test and predicted values")
    # plt.plot(data, label="val")
    plt.plot(real_values, label=caption_for_real_values)
    plt.plot(predicted_values, label=caption_for_predicted_values)
    plt.xlabel("Days")
    plt.ylabel("Close Price")
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
def save_model(model, optimizer, epoch, loss):
    state_dicts = { 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'epoch': epoch, 
                    'loss': loss
                    }
    model_state_dict_path = parameters.model_state_dict_path
    torch.save(state_dicts, model_state_dict_path)