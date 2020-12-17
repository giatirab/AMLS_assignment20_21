# Import libraries
import pandas as pd
import os
import shutil

def count_n_images(path, extension):
    """Returns the number of files within a given folder."""
    list_dir = []
    list_dir = os.listdir(path)
    count = 0
    for file in list_dir:
        if file.endswith(extension):
            count += 1
    return count

def rename_images_infolder(folder, len_train_val, extension):
    """Renames the images in the "_test" img folders in order that they can be moved to the main img folder."""
    
    files = sorted(os.listdir(folder), key = lambda x: int(x[:-4]))

    for index, file in enumerate(files):
        os.rename(os.path.join(folder, file), os.path.join(folder, ''.join([str(index+len_train_val), extension])))

def move_images_fromfolder(folder, destination, extension):
    """Moves image files from one local directory to another."""
    files = os.listdir(folder)
    for f in files:
        if os.path.splitext(f)[1] in (extension):
            shutil.move(folder + f, destination)
            
def add_test_set_tofolder(folder, destination, len_train_val, extension):
    """Renames the whole set of image files in a folder so that they can be moved to another image folder."""
    
    rename_images_infolder(folder, len_train_val, extension)
    move_images_fromfolder(folder, destination, extension)
    

def add_test_set_labels_tofolder(folder, destination, len_train_val, extension, usecols, key_labels_col):
    """Renames the file_name column of the labels.csv file in the test set folder and appends 
       the whole dataframe to the original labels.csv."""
    
    f = lambda x: str(int(os.path.splitext(x)[0]) + len_train_val) + extension
    
    labels_file_0 = pd.read_csv(folder, sep="\t", usecols=usecols)
    labels_file_0[key_labels_col] = labels_file_0[key_labels_col].apply(f)
    labels_file_1 = pd.read_csv(destination, sep="\t", usecols=usecols)
    labels_file_1 = pd.concat([labels_file_1, labels_file_0], ignore_index=True)
    
    return labels_file_1