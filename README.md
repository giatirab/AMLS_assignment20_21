# README

- CODEBASE STRUCTURE

The folder includes a central main.py file which returns the solution to the four assigned tasks (one classifier for each problem). Running this file will print accuracy tables on train, validation and test sets. The file has key dependencies on secondary libraries. These are:

1) _CLASS_ImageManager.py: This script includes a general class which has the purpose of performing the whole set of pre-processing/feature extraction on the dataset. It has some established methods which have very specific purposes (please make use of the help() function if you wish to investigate these methods further):
        
    - count_n_images()
    - load_images_from_folder()
    - plot_image()
    - canny_images()
    - crop_part()
    - resize()

2) _CFG_A.py & _CFG_B.py: These are the configuration files which contain the repository links as well as attributes we want to pass to the ImageManager class object. The user defines a dictionary of specific settings (which body part we want to crop, how to resize the images, etc). Changing the dictionaries will directly change the pre-processing for each assigned task.

3) _LIB_DataProcessing.py: This script is actually operating the pre-processing and is the algorithm which includes all the necessary steps. In case the test set images have not been added to the celeba/img and cartoon_set/img folders, this step is then performed in the background. Similarly happens for the labels.csv file, where the 2 files are concatenated one below the other. The aim is to have only one repository for the celeba images and for the cartoon_set images. The dictionary in _CFG_A.py and _CFG_B.py will contain the information about how to split before fitting and predicting steps.

4) _LIB_AuxiliaryFunctions.py: This python file contains some functions which will move and rename the labels.csv and images from the celeba_test and cartoon_set folders onto celeba and cartoon_set folders. Ultimately, we aim at having the datasets and labels in unique subfolders.

- HOW TO RUN MAIN.PY?

In order to run the four chosen classifiers on the processed datasets and obtain performance scores, please open the _CFG_A.py and _CFG_B.py files and change the directory paths at the top. These include the location of the repository of the "AMLS_20-21_SN17024244" project. The user can subsequently amend the dictionaries directly and see how a different pre-processing will affect the results. The codebase assumes that the images (for both celeba and cartoon_set) are stored in one unique local folder. If this is not the case, the code will automatically take labels.csv and img files from the "test" folders and move these in celeba/img and cartoon_set/img. The A1 and A2 models (by default) are trained on the first 3750 images, validated on the next 1250 and tested on the last 1000 images (as for celeba). Similarly, the B1 and B2 models (by default) are trained on the first 7500 images, validated on the next 2500 and tested on the last 2500 images (as for cartoon_set).

- HOW TO RUN THE JUPYTER NOTEBOOKS IN A1, A2, B1, B2 FOLDERS?

In addition to the .py files described above (which are part of the main.py structure), I have included 4 Jupyter notebooks which have the purpose of presenting the solutions to the project with additional details. They will present the steps that each algorithm is performing and comparing the performance of different classifiers on the same assigned task. The overall structure is simpler then the main.py. Here the Jupyters only depend on _CLASS_ImageManager.py, _CFG_jupyter_A.py and _CFG_jupyter_B.py, which are a mix of libraries and configuration files. To run these files, please go in A1/A2/B1/B2 folders and launch the relevant .ipynb. Then change the path appended (this is where the auxilirary .py files are located) at the top and run the blocks sequentially.

- LIBRARIES REQUIRED

sklearn
cv2
PIL
pandas
numpy
os
shutil
seaborn
matplotlib.pyplot
time


