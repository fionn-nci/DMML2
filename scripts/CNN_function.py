############################################################
# DES: Define a 5 layer CNN which can be adjusted based on the following parameters:
#      - Type of loss function
#      - Type of optimisation
#      - Activation function (default = relu)
############################################################

############################################################
# Libraries:
############################################################

import os
#import scripts.set_working_dir as set_wd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################



############################################################
# Developing 2 models, each with:
# 5 layers
# 5 Max Pooling
# 1 has Gradient Descent Optimisation, 1 has RMSProp
# Input variables:
# - Loss function
# - Optimisation learning rate
# - Activation function
############################################################

# Define your data directories
train_data_dir = '/Users/felixhawksworth/Downloads/cleaned_data/train'
test_data_dir = '/Users/felixhawksworth/Downloads/cleaned_data/test'

########################################
# Gradient Descent Optimisation Model:
#######################################3

def cnn_5_layers_sgd(loss, learning_rate, activation = 'relu'):

    ##################################
    # Define model:
    ##################################

    cnn_model = tf.keras.models.Sequential([

        # Parameters to consider:
        # - Number of layers to NN
        # - Number of filters per NN
        # - Dimensions of filter (kernel)
        # - Max pooling: Reduces the dimesnionality of images by reducing pixels from output of previous layer
        #                Pooling layers are used to reduce the dimensions of the feature maps.
        #                Thus, it reduces the number of parameters to learn and the amount of computation performed in the network.

        # 1st layer (verbose)
        tf.keras.layers.Conv2D(filters = 16,
                               kernel_size = (3, 3),
                               activation = activation,
                               input_shape = (300, 300, 3) # x*x pixels, 3 bytes of colour
                               ),
        tf.keras.layers.MaxPooling2D(2, 2), # each layer will result in half the width x half height
                                            # exports new shape = 150x150
        # 2nd layer:
        tf.keras.layers.Conv2D(32,  (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2), # exports new shape = 75 x 75

        # 3rd layer:
        tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2) , # exports new shape = 32 x 32

        # 4th layer:
        tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 5th layer:
        tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation = activation),  # 512 neuron hidden layer

        # Only 1 output neuron = 'normal' and 1 'pneumonia'
        tf.keras.layers.Dense(1, activation=activation)
    ])

    model_summary = cnn_model.summary()
    print(model_summary)

    sgd = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum=0.0, nesterov=False, name='SGD')

    cnn_model.compile(loss = loss,
                      optimizer = sgd,
                      metrics = ['accuracy'])

    ############################################################
    # Train and Test Model:
    ############################################################

    batch_size = 128
    training_size = 3024
    testing_size = 756
    epochs = 5

    fn_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = fn_steps_per_epoch(training_size)
    test_steps = fn_steps_per_epoch(testing_size)

    # Extract dataset from folder:
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255)

    # get training images
    train_gen = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='binary'
    )

    # get testing images
    test_gen = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='binary'
    )

    # train model
    history = cnn_model.fit(
        train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        validation_data = test_gen,
        validation_steps = test_steps
    )

    model_name_loc = r"/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_" + str(loss) + str(learning_rate) + str(activation)
    cnn_model.save(model_name_loc)

    return model_name_loc

######################################
# RMS Prop Optimisation Model:
######################################

def cnn_5_layers_rmsprop(loss, learning_rate, activation = 'relu'):

    ##################################
    # Define model:
    ##################################

    cnn_model = tf.keras.models.Sequential([

        # Parameters to consider:
        # - Number of layers to NN
        # - Number of filters per NN
        # - Dimensions of filter (kernel)
        # - Max pooling: Reduces the dimesnionality of images by reducing pixels from output of previous layer
        #                Pooling layers are used to reduce the dimensions of the feature maps.
        #                Thus, it reduces the number of parameters to learn and the amount of computation performed in the network.

        # 1st layer (verbose)
        tf.keras.layers.Conv2D(filters = 16,
                               kernel_size = (3, 3),
                               activation = activation,
                               input_shape = (300, 300, 3) # x*x pixels, 3 bytes of colour
                               ),
        tf.keras.layers.MaxPooling2D(2, 2), # each layer will result in half the width x half height
                                            # exports new shape = 150x150
        # 2nd layer:
        tf.keras.layers.Conv2D(32,  (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2), # exports new shape = 75 x 75

        # 3rd layer:
        tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2) , # exports new shape = 32 x 32

        # 4th layer:
        tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 5th layer:
        tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation = 'relu'),  # 512 neuron hidden layer

        # Only 1 output neuron = 'normal' and 1 'pneumonia'
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_summary = cnn_model.summary()
    print(model_summary)

    opt = RMSprop(lr = learning_rate)

    cnn_model.compile(loss = loss,
                      optimizer = opt,
                      metrics = ['accuracy'])

    ############################################################
    # Train and Test Model:
    ############################################################

    batch_size = 256
    training_size = 3024
    testing_size = 756
    epochs = 5

    fn_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = fn_steps_per_epoch(training_size)
    test_steps = fn_steps_per_epoch(testing_size)

    # Extract dataset from folder:
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255)

    # get training images
    train_gen = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='binary'
    )

    # get testing images
    test_gen = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='binary'
    )

    # train model
    history = cnn_model.fit(
        train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        validation_data = test_gen,
        validation_steps = test_steps
    )

    model_name_loc = r"/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_" + str(loss) + str(learning_rate)
    cnn_model.save(model_name_loc)

    return model_name_loc

