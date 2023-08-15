from PIL import Image
import numpy as np
import os
import random
import skimage as sk
from skimage import transform
from skimage.util import random_noise

# Creating local functions to manipulate images
def add_noise(image_array):
    # Add random noise to the image
    return random_noise(image_array)

def flip_horizontal(image_array):
    return np.fliplr(image_array)

def tilt_image(image_array):
    # Random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

# Creating a dictionary of the new functions
available_transformations = {
    'rotate': tilt_image,
    'noise': add_noise,
    'flip_horizontal': flip_horizontal
}

# Define the location of the current training images
folder_path_normal = '/Users/Tommy/Documents/2022College/DataMining2/train/NORMAL'
folder_path_viral = '/Users/Tommy/Documents/2022College/DataMining2/train/tuberculosis'

# Define a limit for the new augmented images
num_files_desired = 2000

# Define new folders for augmented and resized images
new_folder_normal_aug = '/Users/Tommy/Documents/2022College/DataMining2/train/AugData/NORMAL'
new_folder_normal_resized = '/Users/Tommy/Documents/2022College/DataMining2/train/AugDataResized/NORMAL'
new_folder_viral_aug = '/Users/Tommy/Documents/2022College/DataMining2/train/AugData/tuberculosis'
new_folder_viral_resized = '/Users/Tommy/Documents/2022College/DataMining2/train/AugDataResized/tuberculosis'

def process_and_save_images(folder_path, new_folder_aug, new_folder_resized):
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    num_generated_files = 0

    while num_generated_files <= num_files_desired:
        image_path = random.choice(images)
        image_to_transform = sk.io.imread(image_path)
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None

        while num_transformations <= num_transformations_to_apply:
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1

        new_file_path_aug = '%s/augmented_image_%s.jpg' % (new_folder_aug, num_generated_files)
        sk.io.imsave(new_file_path_aug, transformed_image)

        im = Image.open(new_file_path_aug)
        new_width = int(im.width / 4)
        new_height = int(im.height / 4)
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        new_file_path_resized = '%s/resized_image_%s.jpg' % (new_folder_resized, num_generated_files)
        im.save(new_file_path_resized)

        print("Saving", num_generated_files)
        num_generated_files += 1

# Process and save "Normal" images
process_and_save_images(folder_path_normal, new_folder_normal_aug, new_folder_normal_resized)

# Process and save "Viral" images
process_and_save_images(folder_path_viral, new_folder_viral_aug, new_folder_viral_resized)