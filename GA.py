"""
Hyper Parameter:
 1. number of layers
 2. neurons in each layer
 3. dropout rate 
 4. learning rate 
"""
import random
import numpy as np

lr_options = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]


def init_model_params(num_models):
    """
    return a list of size 4, containing the the initial set of paramters for
    the machine learning model. 
    Each element among the four elements will have a length 'num_models'
    
    @rtype (num_layers_collected, num_neurons_layers, dropout_collected, lr_collected)
    """
    num_layers_collected = []
    num_neurons_layers   = []   # Collect the number of neurons in each layer
    dropout_collected    = []
    lr_collected         = []
    for _ in range(num_models):
      num_layers = random.randint(2, 7)
      
      # Collect the number of layers the model shall have 
      num_layers_collected.append(num_layers)
      
      neuron_list = []
      for _ in range(num_layers):
          neuron_list.append(random.randint(50, 500))
      neuron_list[0] = 784 # The first layer has to be the dimension of the input layer
      assert num_layers == len(neuron_list) 
    
      num_neurons_layers.append(neuron_list)
      
      # Select a random dropout between [0,1)
      dropout_collected.append(random.random()) 
      
      # Randomly select a earning rate among 'lr_options'
      lr_collected.append(random.sample(lr_options,  1))
      
    return num_layers_collected, num_neurons_layers, dropout_collected, lr_collected 
      


def get_hyperparam_instance():
    return num_layers_collected, num_neurons_layers, dropout_collected, lr_collected 


