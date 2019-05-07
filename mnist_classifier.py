#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Akshat Kumar Nigam
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from GA import get_hyperparam_instance, init_model_params
torch.manual_seed(7)

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device = 'cpu'
print('Running: ', device)
batch_size = 100

# MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)



class MLP(nn.Module):
    
    def __init__(self, h_sizes, out_size, dropout):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        self.out = nn.Linear(h_sizes[-1], out_size)
        self.dropout = dropout
        
    def forward(self, x):
        dropout_layer = nn.Dropout(self.dropout)
        for layer in self.hidden:
            x = F.relu(dropout_layer(layer(x)))
        output= F.softmax(self.out(x), dim=1)

        return output

    
def display_single_image(image): # Assumed shape: (1, 28, 28)
    plt.imshow(image.reshape((28, 28)), cmap='gray')
    plt.show()
    

def eval_model(model, epoch):
     model.eval()
     for item in test_loader:
        prediction_test = model(item[0].reshape(10000, 28*28).to(device))
        _, indices = prediction_test.max(1)
                    
        indices = indices.cpu().numpy()
        actual = item[1].cpu().numpy()
        num_correct = sum(indices == actual)
                    
        perc_correct = (num_correct / 10000) * 100
     model.train()
     return perc_correct


def train_model(layer_num_neurons, learning_rate, dropout):
    num_epochs = 1  # TODO: Number of epochs must be increased for the final test 
    model = MLP(layer_num_neurons, 10, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, x in enumerate(train_loader):
            predictions = model(x[0].reshape(batch_size, 28*28).to(device)) # (batch_size, output_size) = (1000, 10) 
            loss = criterion(predictions, x[1].to(device))
            # print('Epoch[{}/{}]'.format(epoch, num_epochs),' Batch[{}/{}]'.format(batch_idx, len(train_loader))   ,' Loss: ', loss.item())
              
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
    
        # Implement a learning check...
        perc_correct = eval_model(model, epoch)
        if epoch == num_epochs-1:
            return perc_correct


num_layers_collected, num_neurons_layers, dropout_collected, lr_collected = init_model_params(num_models=5)

def train_multiple_models():
    evaluated_model_metrics = []
    print('We are going to train {} models!'.format(len(num_layers_collected)))
    for i in range(len(num_layers_collected)):
        num_layers, layer_num_neurons, dropout, learning_rate = num_layers_collected[i], num_neurons_layers[i], dropout_collected[i], lr_collected[i][0]
        print('Num Layer: ', num_layers, ' layer_num_neurons:', layer_num_neurons, ' Dropout: ', dropout, ' learning_rate: ', learning_rate)
        assert num_layers == len(layer_num_neurons) # Ensuring proper parameters 
        evaluated_model_metrics.append(train_model(layer_num_neurons, learning_rate, dropout))
    assert len(evaluated_model_metrics) == len(num_layers_collected)
    return evaluated_model_metrics


### THIS CODE SHALL BE ADDED TO 'GA.py'
    
# GET THE BEST ORDERING OF PARAMETERS
evaluated_model_metrics = train_multiple_models()
best_model_indices = list(np.argsort(evaluated_model_metrics))
best_model_indices.reverse()
best_model_params = []  # Contains best model parameters (ranked based on indices)
best_model_metrics = [] # Contains best accuaracies (ranked based on indices ) 
for x in best_model_indices:
    best_model_params.append([num_layers_collected[x], num_neurons_layers[x], dropout_collected[x], lr_collected[x]])
    best_model_metrics.append(evaluated_model_metrics[x])


# Discard all the useless models 
num_models_kept = int(0.2 * len(best_model_params))
if num_models_kept == 1: # TODO: Need this? Just for diversity?
    num_models_kept += 1
best_model_params  = best_model_params[:num_models_kept]
best_model_metrics = best_model_metrics[:num_models_kept]


# BREED THE BEST!
# [6, [784, 315, 350, 310, 377, 104], 0.1284315066903029, [0.0001]
# [2, [784, 314], 0.3078126640432328, [1e-05]]

# How do you breed the number of layers?




# How do you breed the # neurons in a layer




# How do you breed the dropout probability?




# How do you breed the learning rate?
















