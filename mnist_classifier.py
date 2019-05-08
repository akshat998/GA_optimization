#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Akshat Kumar Nigam
"""
import numpy as np
from numpy.random import choice
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from GA import init_model_params, breed_layers, breed_dropout, breed_lr
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
f = open("outputs.txt", "a+")



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


def train_model(layer_num_neurons, learning_rate, dropout, num_epochs):
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



def train_multiple_models(num_layers_collected, num_neurons_layers, dropout_collected, lr_collected, num_epochs):
    evaluated_model_metrics = []
    print('Begining training {} models!'.format(len(num_layers_collected)))
    for i in range(len(num_layers_collected)):
        num_layers, layer_num_neurons, dropout, learning_rate = num_layers_collected[i], num_neurons_layers[i], dropout_collected[i], lr_collected[i][0]
        print('Num Layer: ', num_layers, ' layer_num_neurons:', layer_num_neurons, ' Dropout: ', dropout, ' learning_rate: ', learning_rate)
        assert num_layers == len(layer_num_neurons) # Ensuring proper parameters 
        evaluated_model_metrics.append(train_model(layer_num_neurons, learning_rate, dropout, num_epochs))
    assert len(evaluated_model_metrics) == len(num_layers_collected)
    return evaluated_model_metrics


### GENETIC EVOLUTION LOOP 
    
# Step 1: Get results from the first round of guesses 
num_models = 10
num_layers_collected, num_neurons_layers, dropout_collected, lr_collected = init_model_params(num_models=num_models)
evaluated_model_metrics = train_multiple_models(num_layers_collected, num_neurons_layers, dropout_collected, lr_collected, num_epochs=1)
f.write('Generation results: '+ str(evaluated_model_metrics) + '\n')


# ....  FOR LOOP .................................
num_generations = 3
for i in range(num_generations):
    print('On Generation: ', i+1)
    f.write('On Generation: '+ str(i+1) + '\n')

    # Step 2: Select the best performing models
    best_model_indices = list(np.argsort(evaluated_model_metrics))

    best_model_indices.reverse()
    best_model_params = []  # Contains best model parameters (ranked based on indices)
    best_model_metrics = [] # Contains best accuaracies (ranked based on indices ) 
    for x in best_model_indices:
        best_model_params.append([num_layers_collected[x], num_neurons_layers[x], dropout_collected[x], lr_collected[x]])
        best_model_metrics.append(evaluated_model_metrics[x])
    
    # Discard all the useless models 
    num_models_kept = int(0.5 * len(best_model_params))
    print('Kept: ', num_models_kept)
    if num_models_kept == 1: num_models_kept += 1 # For smaller training cases 
    best_model_params  = best_model_params[:num_models_kept]
    best_model_metrics = best_model_metrics[:num_models_kept]
    
    # Step 3: Add models by breeding 
    num_added_models = num_models - num_models_kept
    bred_models = []
    for i in range(num_added_models+num_models_kept):
        # Select models for breeding: pass 'best_model_metrics' through softmax & Sample two element
        A = list(np.exp(best_model_metrics) / np.sum(np.exp(best_model_metrics), axis=0))
        B = [x for x in range(len(best_model_params))]
        draw = choice(B, 2, A)
        model_choice_1 = best_model_params[draw[0]].copy()
        model_choice_2 = best_model_params[draw[1]].copy()
        if i == 0:
             bred_models.append(model_choice_1)
             bred_models.append(model_choice_2)
    
        # Decide which parameter to breed over (a parameter is selected randomly)
        breed_parameter = random.randint(0, 3) # 0=#layers, 1=#neurons, 2=dropout , 3=lr
        if breed_parameter == 0:
            child = breed_layers(model_choice_1, model_choice_2)
        elif breed_parameter == 1:
            child = breed_layers(model_choice_1, model_choice_2)
        if breed_parameter == 2:
            child = breed_dropout(model_choice_1, model_choice_2)
        if breed_parameter == 3:
            child = breed_lr(model_choice_1, model_choice_2)
        bred_models.append(child)
    print('Total: ', len(bred_models), ' children bred.')
    
    # Step 4: Train bred models and loop. 
    num_layers_collected = []
    num_neurons_layers = []
    dropout_collected = []
    lr_collected = []
    for item in bred_models:
        num_layers_collected.append(item[0])
        num_neurons_layers.append(item[1])
        dropout_collected.append(item[2])
        lr_collected.append(item[3])
        
    evaluated_model_metrics = train_multiple_models(num_layers_collected, num_neurons_layers, dropout_collected, lr_collected, num_epochs=i+2)
    print('Generation best: ', max(evaluated_model_metrics))
    f.write('Generation results: '+ str(evaluated_model_metrics) + '\n')








