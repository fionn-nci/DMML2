############################################################
# DES: Load in the trained LeNet models
# Run on validation dataset
############################################################

############################################################
# Load Libraries:
############################################################

import os
import pandas as pd
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


#try:
#    import scripts.set_working_dir as set_wd
#except:
#    import set_working_dir as set_wd

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

#working_dir = set_wd.set_correct_working_dir()

############################################################
# Get trained models: 5 layers CNNs
############################################################

###########################
# Optimised using SGD:
###########################

# loss function: 'binary_crossentropy'
# - SGD = 0.1, 0.01, 0.001
cnn5_bc_1 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_binary_crossentropy0.1')
cnn5_bc_2 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_binary_crossentropy0.01')
cnn5_bc_3 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_binary_crossentropy0.001')

# loss function: 'mean_squared_error'
# - SGD = 0.1, 0.01, 0.001
cnn5_mse_1 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_mean_squared_error0.1')
cnn5_mse_2 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_mean_squared_error0.01')
cnn5_mse_3 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_mean_squared_error0.001')

# loss function: 'mean_squared_logarithmic_error'
# - SGD = 0.1, 0.01, 0.001
cnn5_msle_1 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_mean_squared_logarithmic_error0.1')
cnn5_msle_2 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_mean_squared_logarithmic_error0.01')
cnn5_msle_3 = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5layer_mean_squared_logarithmic_error0.001')

#################################
# Optimised using RMSprop:
#################################

# loss function: 'binary_crossentropy'
# - RMsProp = 0.1, 0.01, 0.001
cnn5_bc_1_rm = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_binary_crossentropy0.1')
cnn5_bc_2_rm = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_binary_crossentropy0.01')
cnn5_bc_3_rm = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_binary_crossentropy0.001')

# loss function: 'mean_squared_error'
# - RMsProp = 0.1, 0.01, 0.001
cnn5_mse_1_rm = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_mean_squared_error0.1')
cnn5_mse_2_rm = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_mean_squared_error0.01')
cnn5_mse_3_rm = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_mean_squared_error0.001')

# loss function: 'mean_squared_logarithmic_error'
# - RMsProp = 0.1, 0.01, 0.001
cnn5_msle_1_rm = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_mean_squared_logarithmic_error0.1')
cnn5_msle_2_rm = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_mean_squared_logarithmic_error0.01')
cnn5_msle_3_rm = tf.keras.models.load_model(r'/Users/felixhawksworth/Downloads/saved_models/cnn_5lyr_rmsprp_mean_squared_logarithmic_error0.001')

#################################
# Combine all models
#################################

models = [[cnn5_bc_1, "cnn_5layer_binary_crossentropy0.1"], [cnn5_bc_2, "cnn_5layer_binary_crossentropy0.01"], [cnn5_bc_3, "cnn_5layer_binary_crossentropy0.001"],
          [cnn5_mse_1, "cnn_5layer_mean_squared_error0.1"], [cnn5_mse_2, "cnn_5layer_mean_squared_error0.01"], [cnn5_mse_3, "cnn_5layer_mean_squared_error0.001"],
          [cnn5_msle_1, "cnn_5layer_mean_squared_logarithmic_error0.1"], [cnn5_msle_2, "cnn_5layer_squared_logarithmic_error0.01"],
          [cnn5_msle_3, "cnn_5layer_mean_squared_logarithmic_error0.001"],
          [cnn5_bc_1_rm, "cnn_5lyr_rmsprpbinary_crossentropy0.1"], [cnn5_bc_2_rm, "cnn_5lyr_rmsprpbinary_crossentropy0.01"], [cnn5_bc_3_rm, "cnn_5lyr_rmsprpbinary_crossentropy0.001" ],
          [cnn5_mse_1_rm, "cnn_5lyr_rmsprpmean_squared_error0.1"], [cnn5_mse_2_rm, "cnn_5lyr_rmsprpmean_squared_error0.01"], [cnn5_mse_3_rm, "cnn_5lyr_rmsprpmean_squared_error0.001"],
          [cnn5_msle_1_rm, "cnn_5lyr_rmsprpmean_squared_logarithmic_error0.1"], [cnn5_msle_2_rm, "cnn_5lyr_rmsprpmean_squared_logarithmic_error0.01"],
          [cnn5_msle_3_rm, "cnn_5lyr_rmsprpmean_squared_logarithmic_error0.001"] ]

############################################################
# Validate Model: get final results
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

cnn_output = get_validation_results(model_list=models, batch_size=128, target_size=(300,300))
cnn_accuracy = cnn_output[0]
cnn_predictions = cnn_output[1]

############################################################
# Export CSV:
############################################################

df_results = pd.DataFrame()
all_results = []
all_models = []

# CNN 5 layer
for i in cnn_accuracy:
    all_results.append(i[1])
    all_models.append(i[0])

df_results['ACCURACY'] = all_results
df_results['MODEL'] = all_models
df_results1 = df_results.sort_values('ACCURACY')
df_results1 = df_results1.reset_index()
df_results1 = df_results1[['ACCURACY', 'MODEL']]

df_results1.to_csv(r"/Users/felixhawksworth/Downloads/results//cnn5_results.csv", index=False)