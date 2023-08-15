############################################################
# DES: Load in the trained LeNet model
# Run model on validation dataset
############################################################

############################################################
# Load Libraries:
############################################################

import os
import tensorflow as tf
import pandas as pd
import math
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math
import numpy as np

############################################################
# Get trained LeNet models:
############################################################

# loss function: 'binary_crossentropy'
# - SGD = 0.1,  0.01,  0.001
leNet_bc_1 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/LeNet_binary_crossentropy0.1')
leNet_bc_2 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/LeNet_binary_crossentropy0.01')
leNet_bc_3 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/LeNet_binary_crossentropy0.001')

# loss function: 'mean_squared_error'
# - SGD = 0.1,  0.01,  0.001
leNet_mse_1 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/LeNet_mean_squared_error0.1')
leNet_mse_2 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/LeNet_mean_squared_error0.01')
leNet_mse_3 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/LeNet_mean_squared_error0.001')

# loss function: 'mean_squared_logarithmic_error'
# - SGD = 0.1,  0.01,  0.001
leNet_msle_1 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/LeNet_mean_squared_logarithmic_error0.1')
leNet_msle_2 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/LeNet_mean_squared_logarithmic_error0.01')
leNet_msle_3 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/LeNet_mean_squared_logarithmic_error0.001')

models = [[leNet_bc_1, "LeNet_binary_crossentropy0.1"], [leNet_bc_2, "LeNet_binary_crossentropy0.01"], [leNet_bc_3, "LeNet_binary_crossentropy0.001"],
          [leNet_mse_1, "LeNet_mean_squared_error0.1"], [leNet_mse_2, "LeNet_mean_squared_error0.01"], [leNet_mse_3, "LeNet_mean_squared_error0.001"],
          [leNet_msle_1, "LeNet_mean_squared_logarithmic_error0.1"], [leNet_msle_2, "LeNet_mean_squared_logarithmic_error0.01"],
          [leNet_msle_3, "LeNet_mean_squared_logarithmic_error0.001"]]

############################################################
# Validate Models:
############################################################

def get_validation_results(model_list, batch_size, target_size):

    accuracy = []
    predcitions = []

    for i in model_list:

        # load new unseen dataset
        validation_datagen = ImageDataGenerator(rescale=1 / 255)
        val_generator = validation_datagen.flow_from_directory(
            r'/Users/felixhawksworth/Downloads/cleaned_data/validate',
            target_size= target_size,
            batch_size = batch_size,
            class_mode='binary'
        )

        # accuracy summary
        eval_result = i[0].evaluate_generator(val_generator)
        print('Loss rate for validation: ', eval_result[0])
        print('Accuracy rate for validation: ', eval_result[1])

        # get predictions:
        predictions = i[0].predict(val_generator, verbose=1)
        predictions_array = np.array(predictions)
        print(predictions_array.shape)
        predicted_classes = np.argmax(predictions_array, axis=1)

        # save results:
        accuracy.append([i[1], eval_result[1]])
        predcitions.append([i[1], predicted_classes])

    return [accuracy, predcitions]

#########################################
# LeNet:
#########################################

lenet_output = get_validation_results(model_list=models, batch_size=128, target_size=(32,32))
lenet_accuracy = lenet_output[0]
lenet_predictions = lenet_output[1]

############################################################
# Export CSV:
############################################################

df_results = pd.DataFrame()
all_results = []
all_models = []

# Lenet results
for i in lenet_accuracy:
    all_results.append(i[1])
    all_models.append(i[0])

df_results['ACCURACY'] = all_results
df_results['MODEL'] = all_models
df_results1 = df_results.sort_values('ACCURACY')
df_results1 = df_results1.reset_index()
df_results1 = df_results1[['ACCURACY', 'MODEL']]

df_results1.to_csv(r"/Users/felixhawksworth/Downloads/results/leNet_results.csv", index=False)