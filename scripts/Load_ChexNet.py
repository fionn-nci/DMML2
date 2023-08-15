import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
best_model = load_model('chexnet_model.h5')

# Define validation data directory
validation_data_dir = '/Users/Tommy/Documents/2022College/DataMining2/cleaned_data/valid'

# Define input shape and batch size
input_shape = (224, 224, 3)
batch_size = 32

# Create a data generator for validation
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# Evaluate the model on the validation data
loss, accuracy = best_model.evaluate(validation_generator)

print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
