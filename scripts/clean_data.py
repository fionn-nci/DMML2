#########################################################
# DES: Import raw dataset from '.\scripts\input_files' and export cleaned target dataset into 'cleaned_data' folder.
#
#
# BY: Felix Hawksworth
#########################################################

####################
# Load Libraries:
####################

import numpy

import glob
import re
import shutil
from sklearn.model_selection import train_test_split


try:
    import scripts.set_working_directory as set_wd
except:
    import set_working_dir as set_wd

############################
# Set Working Directory:
############################

working_dir = set_wd.set_correct_working_dir()

###########################################
# Import image data and add to new folder:
###########################################