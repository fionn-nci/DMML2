############################################################
# DES: Creating the LeNet CNN and training the model.
# Once trained, export model to working directory.
# BY: Felix Hawksworth
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
from itertools import product

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

#working_dir = set_wd.set_correct_working_dir()

# Define your data directories
train_data_dir = '/Users/felixhawksworth/Downloads/cleaned_data/train'
test_data_dir = '/Users/felixhawksworth/Downloads/cleaned_data/test'

############################################################
# Creating LeNet model:
# - Parameters to change:
# - Optimisation
# - Loss Function
############################################################

#########################################
# Define the combinations of parameters:
#########################################

# Loss functions
loss_fns = ['binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error']

# Optimisation for SGD learning rate:
opts = [0.1,  0.01, 0.001]

# combinations:
combos = list(product(loss_fns, opts))

#########################################
# Define X models (X = len(loss_fns)*len(opts)
#########################################

for i in combos:

    lenet_cnn_model = tf.keras.models.Sequential([

        # 1st layer: 6 filters, kernel 5 x 5, stride = 1, input = 32 x 32
        tf.keras.layers.Conv2D(filters = 6, kernel_size = 5, strides=1, activation = 'relu', input_shape = (32, 32, 3)),

        # 2nd layer: Avg pooling layer
        tf.keras.layers.AveragePooling2D(),

        # 3rd layer: 16 filters, kernel  5 x 5, stride = 1
        tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, strides=1, activation = 'relu'),

        # 4th layer: Avg pooling layer
        tf.keras.layers.AveragePooling2D(),

        # 5th layer: Flatten, fully connected layer / dense:
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),

        # 6th layer:
        tf.keras.layers.Dense(84, activation='relu'),

        # 7th layer: classify as normal / pneumonia
        tf.keras.layers.Dense(1, activation='relu')

    ])

    model_summary = lenet_cnn_model.summary()
    print(model_summary)

    sgd = tf.keras.optimizers.SGD(learning_rate= i[1], momentum=0.0, nesterov=False, name='SGD')

    lenet_cnn_model.compile(loss = i[0],
                            optimizer = sgd,
                            metrics = ['accuracy'])

    ############################################################
    # Train Model:
    ############################################################

    batch_size = 128
    training_size = 3024
    testing_size = 756
    epochs = 5

    fn_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = fn_steps_per_epoch(training_size)
    test_steps = fn_steps_per_epoch(testing_size)

    # Extract dataset from folder:
    train_datagen = ImageDataGenerator(rescale = 1/255)
    test_datagen = ImageDataGenerator(rescale = 1/255)

    # get training images
    train_gen = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (32, 32),
        batch_size = batch_size,
        class_mode = 'binary'
    )

    # get testing images
    test_gen = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (32, 32),
        batch_size  = batch_size,
        class_mode = 'binary'
    )

    # train model
    history = lenet_cnn_model.fit(
        train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        validation_data = test_gen,
        validation_steps = test_steps
    )

    ############################################################
    # Export Model to working Directory:
    ############################################################

    model_name_loc = r"/Users/felixhawksworth/Downloads/saved_models/LeNet_" + str(i[0]) + str(i[1])

    lenet_cnn_model.save(model_name_loc)