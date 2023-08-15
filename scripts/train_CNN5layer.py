#########################################################
# DES: Run multiple CNNs and find optimal results:
# BY: Felix Hawksworth
#########################################################

import os
import scripts.set_working_dir as set_wd
from matplotlib import *

import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
from PIL import Image
from IPython.display import display
from itertools import product

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()

#########################################################
# Import functions
#########################################################

import scripts.CNN_function as cnn_fns

############################################################
# Train models:
# - 2 models defined in cnn_function: 1 has SGD optimisation, other has RMSProp
# - Run 2 models with different combinations of parameters:
# - - Optimisation
# - - Loss function
# Number of models = 2 * 3 * 3
############################################################



#########################################
# Define combinations of paramters:
#########################################

# Loss functions
loss_fns = ['binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error']

# Optimisation for SGD learning rate:
opts = [0.1,  0.01, 0.001]

# combinations:
combos = list(product(loss_fns, opts))

#########################################
# Define X models (X = len(loss_fns)*len(opts)*2)
#########################################

# Train Gradient Descent Algos:
model_paths = []

# CNN is defined in cnn_function.py
for i in combos:
    model_path = cnn_fns.cnn_5_layers_sgd(i[0], i[1],  activation = 'relu')
    model_paths.append(model_path)

# Train RMSProp Algos:
model_paths1 = []

# CNN is defined in cnn_function.py
for i in combos:
    model_path1 = cnn_fns.cnn_5_layers_rmsprop(i[0], i[1],  activation = 'relu')
    model_paths1.append(model_path1)