# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        nodes = []
        nodes.append(input_features)
        if type(hidden_dim) == int:
            nodes.append(hidden_dim)
        elif type(hidden_dim) == list:
            nodes.extend(hidden_dim)
            
        nodes.append(output_dim)
        
        self.module_list = nn.ModuleList()
        for n_in, n_out in zip(nodes[:-1], nodes[1:]):
            self.module_list.append(nn.Linear(n_in, n_out))
            
        self.dropout = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        # define the feedforward behavior
        for layer in self.module_list[:-1]:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        x = self.module_list[-1](x)
        
        return self.sig(x)