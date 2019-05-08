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
      


#def get_hyperparam_instance():
#    return num_layers_collected, num_neurons_layers, dropout_collected, lr_collected 


def breed_layers(model_choice_1, model_choice_2):
    """
    How do you breed the number of layers?
    If the #layers is same, do nothing. If not, int(average)
    Case: if same (if same neurons too, do nothing)
          else diff (at each iteration, randomly sample 1 model & choose num_neurons per iteration)
    Case: if different (at each iteration, randomly sample 1 model & choose num_neurons per iteration)
    
    Note: Droput and learning rate are randomly sampled from either of the models 
    """
    model_choice_1 = model_choice_1.copy()
    model_choice_2 = model_choice_2.copy()
    
    # Randomly sample dropout from one of the models
    if random.randint(0, 1) == 0:
        drop = model_choice_1[2]
    else:
        drop = model_choice_2[2]
        
    # Randomly sample learning rate from one of the models
    if random.randint(0, 1) == 0:
        lr = model_choice_1[3]
    else:
        lr = model_choice_2[3]


    if model_choice_1[0] == model_choice_2[0]:
        if model_choice_1[1] == model_choice_2[1]: # Same number of neurons in each layer
            return [model_choice_1[0], model_choice_1[1], drop, lr].copy()
        else:
            layers = [784]
            for i in range(1, model_choice_1[0]):
                sample_idx = random.randint(0, 1)
                if sample_idx == 0:
                    layers.append(model_choice_1[1][i])
                elif sample_idx == 1:
                    layers.append(model_choice_2[1][i])
            return [model_choice_1[0], layers, drop, lr].copy()
    else:
        num_layers = (model_choice_1[0] + model_choice_2[0]) // 2
        layers = [784]
        for i in range(1, num_layers):
            sample_idx = random.randint(0, 1)
            if sample_idx == 0:
                try:
                    layers.append(model_choice_1[1][i])
                except IndexError:
                    layers.append(model_choice_2[1][i])
            elif sample_idx == 1:
                try:
                    layers.append(model_choice_2[1][i])
                except IndexError:
                    layers.append(model_choice_1[1][i])
        return [num_layers, layers, drop, lr].copy()

                
    


#if mode
def breed_dropout(model_choice_1, model_choice_2):
    """
    How do you breed the dropout probability?
    If same, do nothing. Else, average
    """
    model_choice_1 = model_choice_1.copy()
    model_choice_2 = model_choice_2.copy()
    # Randomly sample learning rate from one of the models
    if random.randint(0, 1) == 0:
        lr = model_choice_1[3]
    else:
        lr = model_choice_2[3]
        
    # Randomly sample layers from one of the models
    if random.randint(0, 1) == 0:
        num_layers = model_choice_1[0]
        neurons    = model_choice_1[1]
    else:
        num_layers = model_choice_2[0]
        neurons    = model_choice_2[1]        
    
    # mutate dropout
    drop = (model_choice_1[2] + model_choice_2[2]) / 2
        
    return [num_layers, neurons, drop, lr].copy()
    

# How do you breed the learning rate?
# If same, do nothing. Else, average
def breed_lr(model_choice_1, model_choice_2):
    """
    How do you breed the dropout probability?
    If same, do nothing. Else, average
    """
    model_choice_1 = model_choice_1.copy()
    model_choice_2 = model_choice_2.copy()
    # Randomly sample dropout rate from one of the models
    if random.randint(0, 1) == 0:
        drop = model_choice_1[2]
    else:
        drop = model_choice_2[2]
        
    # Randomly sample layers from one of the models
    if random.randint(0, 1) == 0:
        num_layers = model_choice_1[0]
        neurons    = model_choice_1[1]
    else:
        num_layers = model_choice_2[0]
        neurons    = model_choice_2[1]        
    
    # mutate dropout
    lr = (model_choice_1[3][0] + model_choice_2[3][0]) / 2
        
    return [num_layers, neurons, drop, [lr]].copy()