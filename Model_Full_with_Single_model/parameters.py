import torch


# dataset parameters
working_dir = './Model_Full_with_Single_model/'
dataset_filePath = "Model_Full_with_Single_model/EURUSD_DATASET/EURUSD_No_Volume_Buy.csv"
rnn_sequence_length = 5
dataloader_batch_size = 1  # should be equal to batch_size in model parameters
train_size_ratio = 0.8
test_size_ratio = 0.2

# model parameters
rnn_architecture = "LSTM"  # RNN LSTM GRU
# batch_size = 1
input_size = 4  # number of features in every samples
sequence_length = 5  #aggregation 
first_hidden_size = 128
last_hidden_size = 64
num_output = 4
rnn_act_func = torch.tanh
embedding_act_func = torch.relu # Linear activation function: None otherwise: torch.relu or torch.sigmoid
# embedding_act_func = torch.

# saving model or loading it
model_state_dict_path = f"Model_Full_with_Single_model/model_parameters/model_with_general_checkpoint_{rnn_architecture}"
load_model = False


# model tuning parameters
lr = 0.001  # learning rate
epochs = 100


loss = 0
epoch = 0