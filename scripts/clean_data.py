#########################################################
# DES: Import raw dataset from '.\scripts\input_files'
# Export cleaned dataset into 'cleaned_data' folder.
#
#
# BY: Felix Hawksworth
#########################################################

####################
# Load Libraries:
####################

import numpy
import os
import glob
import re
import shutil
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from PIL import Image

try:
    import scripts.set_working_directory as set_wd
except:
    import set_working_dir as set_wd

############################
# Set Working Directory:
############################

working_dir = set_wd.set_working_directory()

###########################################
# Import image data and add to new folder:
###########################################

image_dir_normal = '/Users/felixhawksworth/Downloads/Input_images/Normal/'
image_dir_TB = '/Users/felixhawksworth/Downloads/Input_images/Tuberculosis/'


# Find PNG files in the directory
all_files_norm = glob.glob(os.path.join(image_dir_normal, '*.png'))
all_files_TB = glob.glob(os.path.join(image_dir_TB, '*.png'))

# Testing if images read in:

if not all_files_TB:
    print("No PNG files found in the directory.")
else:
    for png_file in all_files_TB:
        try:
            print("Processing:", png_file)
            image = Image.open(png_file)
            plt.imshow(image)
            plt.title(png_file)
            plt.show()
        except Exception as e:
            print(f"Error processing {png_file}: {e}")

print("Processing complete.")


###############


#######################################
# Export new folder: Cleaned data:
# Test, Train, Validate: Each with Normal and Viral datasets loaded.
#######################################

# Get test and train folders:
test_train_normal = all_files_norm[0:int((len(all_files_norm)*0.9))]
test_train_viral = all_files_TB[0:int((len(all_files_TB)*0.9))]

normal_train, normal_test = train_test_split(test_train_normal, test_size=0.2, random_state=0)
viral_train, viral_test = train_test_split(test_train_viral, test_size=0.2, random_state=0)

for i in normal_train:
    shutil.copy(i, r'/Users/felixhawksworth/Downloads/cleaned_data/train/normal')

for i in normal_test:
    shutil.copy(i, r'/Users/felixhawksworth/Downloads/cleaned_data/test/normal')

for i in viral_train:
    shutil.copy(i, r'/Users/felixhawksworth/Downloads/cleaned_data/train/viral')

for i in viral_test:
    shutil.copy(i, r'/Users/felixhawksworth/Downloads/cleaned_data/test/viral')

# Get validate set:
validate_normal = all_files_norm[int((len(all_files_norm)*0.9)):len(all_files_norm)]
validate_viral = all_files_TB[int((len(all_files_TB) * 0.9)):len(all_files_TB)]

for i in validate_normal:
    shutil.copy(i, r'/Users/felixhawksworth/Downloads/cleaned_data/validate/normal')

for i in validate_viral:
    shutil.copy(i, r'/Users/felixhawksworth/Downloads/cleaned_data/validate/viral')

print("Summary: ")
print("Normal test: ", len(normal_test))
print("Normal train: ", len(normal_train))
print("Normal validate: ", len(validate_normal))
print("TB test: ", len(viral_test))
print("TB train: ", len(viral_train))
print("TB validate: ", len(validate_viral))

print("Filtered raw dataset to new folder: \cleaned_data ")