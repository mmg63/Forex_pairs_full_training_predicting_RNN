#%%
import torch
import numpy as np
from RnnClass import MyRNN
import matplotlib.pyplot as plt
# from custom_dataset import Forex_Dataset
from dataset_with_normalized_Holidays import Forex_train_Dataset, Forex_test_Dataset
from torch.utils.data import DataLoader
from parameters import *
from time import time, ctime
from extended_functions_full_candlestick import *
import mplfinance as mpl

# create an object from the model and define optimizers and backpropagation functions
torch.manual_seed(int(time()))
model = MyRNN(rnn_architecture=rnn_architecture, 
                batch_size=dataloader_batch_size, 
                input_size=input_size, 
                seq_length=sequence_length, 
                num_outputs=num_output,
                embed_activation_function=embedding_act_func, 
                first_stack_hidden_size=first_hidden_size,
                last_stack_hidden_size=last_hidden_size, 
                rnn_activation_function=rnn_act_func
            )

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()
print(f'\x1b[6;30;42m Model_Close {model.__repr__()} \x1b[0m')

# storing losses and accuracies
train_loss = []

loss_Open = 0


# train of load the model
if load_model:
    # loading model and set parameters
    checkpoint = torch.load(model_state_dict_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss_Open = checkpoint["loss"]
 
else:

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

    model.train()
    
    for epoch in range(epochs):
        # tracking time with time moodule.
        time_satrt = time()

        # initializing loss and optimizer for the next run
        optimizer.zero_grad()
        loss = 0
        
        # feed whole samples into the model
        batches = iter(loader)
        for batch_idx in range(len(loader)-4):
            # samples, target_Close = next(batches)
            samples, target = next(batches)
            
            _, _, y_hat = model(samples)

            loss += criterion(y_hat, target)
        
        # append losses for plot
        train_loss.append(loss.item())

        # calculate backpropagation and optimize losses 
        loss.backward()
        optimizer.step()
        
        time_end = time()
        
        if (epoch % 10 == 0):
            # save_single_model(model_Close=model_Close, Optimizer_Close=optimizer_Close, epoch=epoch, loss_Close=loss_Close)
            save_model(model, optimizer, epoch, loss)
            print("epoch : {}, loss : {}, at epoch time of {}"
                    .format(epoch, loss, (time_end - time_satrt)))
            # print("epoc : {}, loss_Close : {}, at epoch teime of {}"
            #         .format(epoch, loss_Close, (time_end - time_satrt)))
        if (epoch == epochs - 1):
            # save_single_model(model_Close=model_Close, Optimizer_Close=optimizer_Close, epoch=epoch, loss_Close=loss_Close)
            save_model(model, optimizer, epoch, loss)
    
    plot_acc_loss(train_loss, "train loss")

#%% test phase
model.eval()

test_dataset = Forex_test_Dataset(dataset_filePath,
                                transform=None, 
                                seq_lenght=sequence_length,
                                train_size_ratio=test_size_ratio)
print(f"length of the dataset is {len(test_dataset)}")

# load dataset based on days and its followed weekdays
test_loader = DataLoader(test_dataset, batch_size=dataloader_batch_size, 
                        shuffle=False, drop_last=True, ) 
test_value = []

test_Pred = []

test_batches = iter(test_loader)
for batch_idx in range(len(test_loader)-4):
    # samples, target_Close = next(test_batches)

    samples, target = next(test_batches)
    _, _, y_pred = model(samples)
    
    test_value.append(target)
    
    test_Pred.append(y_pred.item())

# plot test prices as cangles
# predicted_price_to_csv(test_Pred_Open, test_Pred_High, test_Pred_Low, test_Pred_Close, plot_chart=True)

plot_test_values_predicted(test_value, test_Pred,"test real value", "test predicted value", y_label="Values")
# plt.savefig(f"ModelBasedOnOHLC/plot/Open_at_{ctime()}", dpi=300)

plt.show()
print("The end.")

