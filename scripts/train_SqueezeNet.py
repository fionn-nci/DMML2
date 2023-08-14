############################################################
# Libraries:
############################################################

from tensorflow.keras.layers import Input, Conv2D, Concatenate, \
    MaxPool2D, GlobalAvgPool2D, Activation
from keras.models import Model
import scripts.set_working_dir as set_wd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import math
from itertools import product


#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()

############################################################
# Build model:
# Define function for building a 5 layer CNN with
# - requires the following inputs:
# - - Input Shape
# - - Number of Classes
# - - Activation function (default = relu)
############################################################

def squeezenet(input_shape, n_classes):
    def fire(x, fs, fe):
        s = Conv2D(fs, 1, activation='relu')(x)
        e1 = Conv2D(fe, 1, activation='relu')(s)
        e3 = Conv2D(fe, 3, padding='same', activation='relu')(s)
        output = Concatenate()([e1, e3])
        return output

    input = Input(input_shape)

    x = Conv2D(96, 7, strides=2, padding='same', activation='relu')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = fire(x, 16, 64)
    x = fire(x, 16, 64)
    x = fire(x, 32, 128)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = fire(x, 32, 128)
    x = fire(x, 48, 192)
    x = fire(x, 48, 192)
    x = fire(x, 64, 256)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = fire(x, 64, 256)
    x = Conv2D(n_classes, 1)(x)
    x = GlobalAvgPool2D()(x)

    output = Activation('softmax')(x)

    model = Model(input, output)
    return model


############################################################
# Model Summary:
############################################################

INPUT_SHAPE = 224, 224, 3
N_CLASSES = 2

model = squeezenet(INPUT_SHAPE, N_CLASSES)
model.summary()

# plot_model(model)


#########################################
# Define combinations of parameters:
#########################################

# Loss functions
loss_fns = ['binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error']

# Optimisation for SGD learning rate:
opts = [0.1, 0.01, 0.001]

# combinations:
combos = list(product(loss_fns, opts))

for i in combos:
    ############################################################
    # Define Constants:
    ############################################################

    batch_size = 128
    training_size = 2148
    testing_size = 2686 - training_size
    epochs = 5

    fn_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = fn_steps_per_epoch(training_size)
    test_steps = fn_steps_per_epoch(testing_size)

    ############################################################
    # Extract Data:
    ############################################################

    # Extract dataset from folder:
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255)

    # get training images
    train_gen = train_datagen.flow_from_directory(
        r'.\cleaned_data\train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    # get testing images
    test_gen = test_datagen.flow_from_directory(
        r'.\cleaned_data\test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    ############################################################
    # Train model:
    ############################################################

    sgd = tf.keras.optimizers.SGD(learning_rate=i[1], momentum=0.0)

    model.compile(
        loss=i[0],
        optimizer=sgd,
        metrics=['accuracy']
    )

    history = model.fit(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=test_gen,
                        validation_steps=test_steps
                        )

    model_name_loc = r".\saved_models\Squeeze_" + str(i[0]) + str(i[1])

    model.save(model_name_loc)