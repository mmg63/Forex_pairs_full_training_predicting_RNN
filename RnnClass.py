# from black import out
import torch
from torch import nn
from torch.nn import RNN, LSTM, GRU
from torch import Tensor


# build an Recurrent Neural Network based on specified architecture.
class MyRNN(nn.Module):
    r"""
    This class if for creating a RNN model for time series.
    
    This class is produced two RNN stacked behind eachother

    Args:
    -------
    `input_size` : the number of features fet to one RNN cell,
    `hidden_size` : the number of features extracted as output from one RNN cell,
    `batch_first` : determine the batch in input samples, default is True,
    `seq_length` : the number of sequences in each time step for RNN
    `rnn_architecture` : three possible options: "RNN", "LSTM" or "GRU", default is RNN
    `rnn_activation_function` : activation function using for RNN cells. defulat is tanh. Yet we can use relu
    """
    
    def __init__(self, 
                rnn_architecture="RNN", 
                input_size:int = 1, 
                seq_length:int = 5,
                batch_size:int = 32,
                first_stack_hidden_size:int = 128,
                last_stack_hidden_size:int = 64,
                num_outputs:int = 1, # after the last hidden size for embedding to get the last output.
                rnn_activation_function=torch.nn.Tanh,
                embed_activation_function = torch.relu
                ) -> None:
        """
        Args:
        -----
        `rnn_architecture` : string: "RNN" "LSTM" "GRU". Default is "RNN".
        `batch_size` : number of batches in each run of feeding data to the RNN model.
        `input_size` : number of inputs for each RNN cell.
        `seq_length` : number of RNN cells stacking behid each other.
        `num_output` : number of cells after embedding layer to get the last and desirable output.
        `embed_activation_function` : embedding activation function.
        `fitst_stack_hidden_size` : number of neurons for the first hidden size.
        `last_stack_hidden_size` : number of neurons output from the last rnn cell and 
                                    get ready to use as input layer for embedding layer.
        `rnn_activation_function` : activation function used in RNN cells.

        """
        super().__init__()

        self.num_sequences = seq_length
        self.batch_size = batch_size
        self.num_features = input_size
        self.first_stack_hidden_size = first_stack_hidden_size
        self.last_stack_hidden_size = last_stack_hidden_size
        self.num_outputs = num_outputs
        self.rnn_activation_function = rnn_activation_function
        self.embed_act_func = embed_activation_function
    

        if rnn_architecture == "RNN":
            self.first_rnn_layer = RNN(input_size=self.num_features, 
                                        hidden_size=self.first_stack_hidden_size,
                                        # nonlinearity = self.rnn_activation_function,
                                        batch_first = True,
                                        bias=True)
            self.second_rnn_layer = nn.RNN(input_size=self.first_stack_hidden_size, 
                                        hidden_size=self.last_stack_hidden_size,
                                        # nonlinearity = self.rnn_activation_function,
                                        batch_first = True,
                                        bias=True)
        
        elif rnn_architecture == "LSTM":
            self.first_rnn_layer = LSTM(input_size=self.num_features, 
                                        hidden_size=self.first_stack_hidden_size,
                                        # nonlinearity = self.rnn_activation_function,
                                        batch_first = True,
                                        bias=True)
            self.second_rnn_layer = LSTM(input_size=self.first_stack_hidden_size, 
                                        hidden_size=self.last_stack_hidden_size,
                                        # nonlinearity = self.rnn_activation_function,
                                        batch_first = True,
                                        bias=True)
        elif rnn_architecture == "GRU":
            self.first_rnn_layer = GRU(input_size=self.num_features, 
                                        hidden_size=self.first_stack_hidden_size,
                                        # nonlinearity = self.rnn_activation_function,
                                        batch_first = True,
                                        bias=True)
            self.second_rnn_layer = nn.GRU(input_size=self.first_stack_hidden_size, 
                                        hidden_size=self.last_stack_hidden_size,
                                        # nonlinearity = self.rnn_activation_function,
                                        batch_first = True,
                                        bias=True)
            
        # embedding layer
        self.embedding = nn.Linear(in_features=self.last_stack_hidden_size,out_features=self.num_outputs,
                                        bias=True)

    
    def forward(self, x):
        """
        return hidden1, hidden2, ouput of the model.
        """
        x, hidden1 = self.first_rnn_layer(x)
        x, hidden2 = self.second_rnn_layer(x)
        # print(f'shape of output of the last rnn layers : {x.shape}')
        # print(f'output of the last rnn architecture for feeding to the embedding layer is : {x[0,-1,:].shape}')
        y_hat = self.embedding(x[0,-1,:])  # output of the last rnn architecture for feeding to the embedding layer
        if self.embed_act_func != None:
            y_hat = self.embed_act_func(y_hat)

        return hidden1, hidden2, y_hat