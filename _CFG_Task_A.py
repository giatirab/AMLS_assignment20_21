# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import shutil

folder = "/Users/macbookpro/UCL - MSc Integrated Machine Learning Systems/Y1/Applied Machine Learning I/Final Assignment/AMLS_20-21_SN17024244/Datasets/celeba/"
test_folder = "/Users/macbookpro/UCL - MSc Integrated Machine Learning Systems/Y1/Applied Machine Learning I/Final Assignment/AMLS_20-21_SN17024244/Datasets/celeba_test/"
sub_folder = folder + "img/"
sub_test_folder = test_folder + "img/"
extension = ".jpg"

def load_images_label_csv(label):
    """Loads the labels csv file and return the column specified as argument."""
    
    FILE_NAME = "labels.csv"
    SEPARATOR = "\t"
    USECOLS = ["img_name", "gender", "smiling"]
    
    labels_file = pd.read_csv(folder + FILE_NAME, sep=SEPARATOR, usecols=USECOLS)
    labels = labels_file[label].values
    
    return labels

def train_test_confusion_matrix(y_train, y_pred_train, y_test, y_pred_test):
    """Creates the confusion matrix on both train and test sets based on the classification outcome."""
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    mat1 = confusion_matrix(y_train, y_pred_train)
    mat2 = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(mat1.T, square=True, annot=True, fmt='d', cbar=False, cmap=plt.cm.Blues, linewidths=0.2, ax=ax1)
    sns.heatmap(mat2.T, square=True, annot=True, fmt='d', cbar=False, cmap=plt.cm.Blues, linewidths=0.2, ax=ax2)
    ax1.set_title('Confusion Matrix for Train set', y=-0.1)
    ax2.set_title('Confusion Matrix for Test set', y=-0.1)
    plt.show()
    
def move_images_fromfolder(folder, destination, extension):
    """Moves image files from one local directory to another."""
    files = os.listdir(folder)
    for f in files:
        if os.path.splitext(f)[1] in (extension) and int(os.path.splitext(f)[0]) >= 5000:
            shutil.move(folder + f, destination)

def count_n_images(path, extension):
    """Returns the number of files within a given folder."""
    list_dir = []
    list_dir = os.listdir(path)
    count = 0
    for file in list_dir:
        if file.endswith(extension):
            count += 1
    return count

        