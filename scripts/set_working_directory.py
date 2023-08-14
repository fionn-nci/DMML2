############################################################
# DES: Ensures that all scripts in project are ran from the correct location:
# Correct location = '.\scripts'
# BY: Felix Hawksworth
############################################################

#################
# Load Libraries:
#################

import os

#########################################################
# Function for Setting Working Directory:
# Ensures RELATIVE working directory for easy replication
# Ensures the users can read data in via  Windows or UNIX folders
# NOTE: Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################


def set_working_directory():

    current_dir = os.getcwd()
    print("Current directory: ", current_dir)

    if current_dir[len(current_dir)-7:len(current_dir)] != 'scripts':
        try:
            os.chdir(r"")
            print("Changing working directory to: ", os.getcwd())
            print("New working directory: ", os.getcwd())
        except:
            print(r"Can't find .\scripts folder, will try '/scripts' instead (Windows v UNIX) ")

            try:
                os.chdir(r"")
                print("Changing working directory to: ", os.getcwd())
                print("New working directory: ", os.getcwd())
            except:
                print(r"Still can't find correct directory, continuing script anyway")
    else:
        print("Working directory already correct: ", os.getcwd())

    working_dir = os.getcwd()

    return working_dir