#%%
import torch
import numpy as np
from RnnClass import MyRNN
import matplotlib.pyplot as plt
# from custom_dataset import Forex_Dataset
from custom_dataset_train_test_split import Forex_train_Dataset, Forex_test_Dataset
from torch.utils.data import DataLoader
from parameters_for_four_price_prediction import *
from time import time
from extended_functions_full_candlestick import *


# create an object from the model and define optimizers and backpropagation functions
model_Open = MyRNN(rnn_architecture=rnn_architecture, 
                batch_size=dataloader_batch_size, 
                input_size=input_size, 
                seq_length=sequence_length, 
                num_outputs=num_output,
                embed_activation_function=embedding_act_func, 
                first_stack_hidden_size=first_hidden_size,
                last_stack_hidden_size=last_hidden_size, 
                rnn_activation_function=rnn_act_func
            )
model_High = MyRNN(rnn_architecture=rnn_architecture, 
                batch_size=dataloader_batch_size, 
                input_size=input_size, 
                seq_length=sequence_length, 
                num_outputs=num_output,
                embed_activation_function=embedding_act_func, 
                first_stack_hidden_size=first_hidden_size,
                last_stack_hidden_size=last_hidden_size, 
                rnn_activation_function=rnn_act_func
            )
model_Low = MyRNN(rnn_architecture=rnn_architecture, 
                batch_size=dataloader_batch_size, 
                input_size=input_size, 
                seq_length=sequence_length, 
                num_outputs=num_output,
                embed_activation_function=embedding_act_func, 
                first_stack_hidden_size=first_hidden_size,
                last_stack_hidden_size=last_hidden_size, 
                rnn_activation_function=rnn_act_func
            )
model_Close = MyRNN(rnn_architecture=rnn_architecture, 
                batch_size=dataloader_batch_size, 
                input_size=input_size, 
                seq_length=sequence_length, 
                num_outputs=num_output,
                embed_activation_function=embedding_act_func, 
                first_stack_hidden_size=first_hidden_size,
                last_stack_hidden_size=last_hidden_size, 
                rnn_activation_function=rnn_act_func
            )

optimizer_Open = torch.optim.Adam(model_Open.parameters(), lr=lr)
optimizer_High = torch.optim.Adam(model_High.parameters(), lr=lr)
optimizer_Low = torch.optim.Adam(model_Low.parameters(), lr=lr)
optimizer_Close = torch.optim.Adam(model_Close.parameters(), lr=lr)

criterion_Open = torch.nn.MSELoss()
criterion_High = torch.nn.MSELoss()
criterion_Low = torch.nn.MSELoss()
criterion_Close = torch.nn.MSELoss()

print(f'\x1b[6;30;42m Model_open {model_Open.__repr__()} \x1b[0m')
print(f'\x1b[6;30;42m Model_High {model_High.__repr__()} \x1b[0m')
print(f'\x1b[6;30;42m Model_Low {model_Low.__repr__()} \x1b[0m')
print(f'\x1b[6;30;42m Model_Close {model_Close.__repr__()} \x1b[0m')


# import dataset from custom dataset
ds = Forex_train_Dataset(dataset_filePath,
                    transform=None, 
                    seq_lenght=sequence_length,
                    train_size_ratio=train_size_ratio)
print(f"length of the dataset is {len(ds)}")


# load dataset based on days and its followed weekdays
loader = DataLoader(ds, batch_size=dataloader_batch_size, 
                    shuffle=False, drop_last=True, ) 
                    # num_workers=1, pin_memory=True,
                    # multiprocessing_context=8)


# storing losses and accuracies
train_loss_Open = []
train_loss_High = []
train_loss_Low = []
train_loss_Close = []

loss_Open = 0
loss_High = 0
loss_Low = 0
loss_Close = 0


# train of load the model
if load_model:
    # loading model and set parameters
    checkpoint = torch.load(model_state_dict_path)
    model_Open.load_state_dict(checkpoint["model_Open_state_dict"])
    model_High.load_state_dict(checkpoint["model_High_state_dict"])
    model_Low.load_state_dict(checkpoint["model_Low_state_dict"])
    model_Close.load_state_dict(checkpoint["model_Close_state_dict"])
    optimizer_Open.load_state_dict(checkpoint["optimizer_Open_state_dict"])
    optimizer_High.load_state_dict(checkpoint["optimizer_High_state_dict"])
    optimizer_Low.load_state_dict(checkpoint["optimizer_Low_state_dict"])
    optimizer_Close.load_state_dict(checkpoint["optimizer_Close_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss_Open"]
    loss = checkpoint["loss_High"]
    loss = checkpoint["loss_Low"]
    loss = checkpoint["loss_Close"]

    print("epoc : {}, loss :{}".format(epoch, loss_Open))
    print("epoc : {}, loss :{}".format(epoch, loss_High))
    print("epoc : {}, loss :{}".format(epoch, loss_Low))
    print("epoc : {}, loss :{}".format(epoch, loss_Close))
else:
    model_Open.train()
    model_High.train()
    model_Close.train()
    model_Low.train()
    for epoch in range(epochs):
        # tracking time with time moodule.
        time_satrt = time()

        # initializing loss and optimizer for the next run
        optimizer_Open.zero_grad()
        optimizer_High.zero_grad()
        optimizer_Low.zero_grad()
        optimizer_Close.zero_grad()
        loss_Open = 0
        loss_High = 0
        loss_Low = 0
        loss_Close = 0

        # feed whole samples into the model
        batches = iter(loader)
        for batch_idx in range(len(loader)-4):
            samples, target = next(batches)
            h1, h2, y_hat_Open = model_Open(samples)
            h1, h2, y_hat_High = model_High(samples)
            h1, h2, y_hat_Low = model_Low(samples)
            h1, h2, y_hat_Close = model_Close(samples)
            loss_Open += criterion_Open(y_hat_Open, target)
            loss_High += criterion_High(y_hat_High, target)
            loss_Low += criterion_Low(y_hat_Low, target)
            loss_Close += criterion_Close(y_hat_Close, target)
        
        # append losses for plot
        train_loss_Open.append(loss_Open.item())
        train_loss_High.append(loss_High.item())
        train_loss_Low.append(loss_Low.item())
        train_loss_Close.append(loss_Close.item())
        # calculate backpropagation and optimize losses 
        loss_Open.backward()
        loss_High.backward()
        loss_Low.backward()
        loss_Close.backward()
        optimizer_Open.step()
        optimizer_High.step()
        optimizer_Low.step()
        optimizer_Close.step()
        
        time_end = time()
        
        if (epoch % 10 == 0):
            save_model(model_Open, model_High, model_Low, model_Close, 
                        optimizer_Open, optimizer_High, optimizer_Low, optimizer_Close,
                        epoch, loss_Open, loss_High, loss_Low, loss_Close)
            print(f'epoch: {epoch}, time: %.4f , loss_Open: %.4f' % ((time_end - time_satrt), loss_Open.item()))
            print(f'epoch: {epoch}, time: %.4f , loss_High: %.4f' % ((time_end - time_satrt), loss_High.item()))
            print(f'epoch: {epoch}, time: %.4f , loss_Low: %.4f' % ((time_end - time_satrt), loss_Low.item()))
            print(f'epoch: {epoch}, time: %.4f , loss_Close: %.4f' % ((time_end - time_satrt), loss_Close.item()))
        
        if (epoch == epochs - 1):
            save_model(model_Open, model_High, model_Low, model_Close, 
                        optimizer_Open, optimizer_High, optimizer_Low, optimizer_Close,
                        epoch, loss_Open, loss_High, loss_Low, loss_Close)
    plot_acc_loss(train_loss_Open, "train loss_Open")
    plot_acc_loss(train_loss_High, "train l_Highoss")
    plot_acc_loss(train_loss_Low, "train l_Lowoss")
    plot_acc_loss(train_loss_Close, "train lo_Closess")

#%% test phase
model_Open.eval()
model_High.eval()
model_Low.eval()
model_Close.eval()

test_dataset = Forex_test_Dataset(dataset_filePath,
                                transform=None, 
                                seq_lenght=sequence_length,
                                train_size_ratio=test_size_ratio)
print(f"length of the dataset is {len(test_dataset)}")

# load dataset based on days and its followed weekdays
test_loader = DataLoader(test_dataset, batch_size=dataloader_batch_size, 
                        shuffle=False, drop_last=True, ) 
test_batches = iter(loader)
test_value = []
test_Pred_Open = []
test_Pred_High = []
test_Pred_Low = []
test_Pred_Close = []
for batch_idx in range(len(test_loader)-4):
    samples, target = next(test_batches)
    _, _, y_pred_Open = model_Open(samples)
    _, _, y_pred_High = model_High(samples)
    _, _, y_pred_Low = model_Low(samples)
    _, _, y_pred_Close = model_Close(samples)
    test_value.append(target)
    
    test_Pred_Open.append(y_pred_Open.item())
    test_Pred_High.append(y_pred_High.item())
    test_Pred_Low.append(y_pred_Low.item())
    test_Pred_Close.append(y_pred_Close.item())

plot_test_values_predicted(test_value[-11:], test_Pred_Close[-10:],"test real value", "test predicted value" )
# plot_test_values_predicted(test_value, test_acc[1:],"test real value", "test predicted value" )

# plot_acc_loss(test_acc, "test predicted value")
# plot_acc_loss(test_value, "test real value")

print("The end.")
