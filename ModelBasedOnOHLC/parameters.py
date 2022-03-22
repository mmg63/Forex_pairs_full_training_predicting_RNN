import torch

# dataset parameters
dataset_filePath = "EURUSD_DATASET/EURUSD_1.02.2007_26.02.2022_Full_prediction.csv"
rnn_sequence_length = 4
dataloader_batch_size = 1  # should be equal to batch_size in model parameters
train_size_ratio = 0.8
test_size_ratio = 0.2

# model parameters
rnn_architecture = "LSTM"  # RNN LSTM GRU
# batch_size = 1
input_size = 6
sequence_length = 4
first_hidden_size = 128
last_hidden_size = 64
num_output = 1
rnn_act_func = torch.tanh
embedding_act_func = torch.relu # Linear activation function: None otherwise: torch.relu or torch.sigmoid
# embedding_act_func = torch.

# saving model or loading it
model_state_dict_path = f"model_parameters/model_with_general_checkpoint_{rnn_architecture}"
load_model = True


# model tuning parameters
lr = 0.0005  # learning rate
epochs = 400


loss = 0
epoch = 0