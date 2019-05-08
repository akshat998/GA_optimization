# GA_optimization

This repository encourages the use of Genetic Algorithms for hyperparameter optimization. /

For testing, a classifier for MNIST digits is implemented. /

The motivation is to provide an alternative stratergy to Baysean Optimization. With more that 20 hyperparameters, baysean optimization becomes infeasible. 

# Hyperparameters

The code attempts to optimize paramters:

- Number of layers in the model
- The number of neurons in each layer
- The dropout rate
- The learning rate 

After generation 0 (for which parameters are randomly selected), the top performing models are selected, and multiple children are bred using randomness (modifying randomly selected parameter from above). 


# Installation Requirements

Before running the code, please ensure you have the following:

- [Python 3.0 or up](https://www.python.org/download/releases/3.0/)
- [Pytorch v0.4.1](https://pytorch.org/)

# Getting Started

For a quick start, please ensure the following.

- Clone the repository:

  In an appropriate directory run the following command on your Terminal:

  `git clone https://github.com/akshat998/GA_optimization.git`

- Make sure you `cd` into the right directory.

  `cd GA_optimization/`

- Train your models:

  To initiate training for several MNIST classifiers :

  ` python3 mnist_classifier.py`

  For quicker training, the following variables in `mnist_classifier.py` can be modified:
   - num_models     : Indicating the number of models in the starting generation
   - num_generations: The number of breeding generations

  Results for each generation of models is saved in 

   `outputs.txt`



# Questions, problems?

Make a github issue 😄. Please be as clear and descriptive as possible. Please feel free to reach
out in person: (akshat[DOT]nigam[AT]mail[DOT]utoronto[DOT]ca)

