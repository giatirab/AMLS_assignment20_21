import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

folder = "/Users/macbookpro/UCL - MSc Integrated Machine Learning Systems/Y1/Applied Machine Learning I/Final Assignment/AMLS_20-21_SN17024244/Datasets/dataset_AMLS_20-21/cartoon_set/"
sub_folder = folder + "img/"
extension = ".png"

def load_images_label_csv(label):
    
    FILE_NAME = "labels.csv"
    SEP = "\t"
    USECOLS = ["eye_color", "face_shape", "file_name"]
    
    labels_file = pd.read_csv(folder + FILE_NAME, sep=SEP, usecols=USECOLS)
    labels = labels_file[label].values
    
    return labels

def train_test_confusion_matrix(y_train, y_pred_train, y_test, y_pred_test):
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    mat1 = confusion_matrix(y_train, y_pred_train)
    mat2 = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(mat1.T, square=True, annot=True, fmt='d', cbar=False, cmap=plt.cm.Blues, linewidths=0.2, ax=ax1)
    sns.heatmap(mat2.T, square=True, annot=True, fmt='d', cbar=False, cmap=plt.cm.Blues, linewidths=0.2, ax=ax2)
    ax1.set_title('Confusion Matrix for Train set', y=-0.1)
    ax2.set_title('Confusion Matrix for Test set', y=-0.1)
    plt.show()