# Import libraries
import _CLASS_ImageManager as mgr
import _LIB_AuxiliaryFunctions as af
import _CFG_A as cfga
import _CFG_B as cfgb
import pandas as pd

def preprocessing(settings, verbose):
    """Creates an instance of ImageManager and uses it to load images and
       applies preprocessing as specified in the settings argument. The 
       settings must contain:
        
           folder    : The folder containing the images
           extension : The image's file extension, such as (.jpg or .png)
           color     : The images's colour used to process the images
           bodypart  : A body part assigned for cropping
           size      : The output size of the images
           squaring  : If True, pads the rectangle around the bodypart to become a square
           x_offsets, y_offsets : Tuples representing how much we should shift the
                               endpoints of the rectangle enclosing the bodypart.
                               Example: 
                                   x_offsets, y_offsets = (1/4, -1/4), (0,0)
                                   This would crop the image starting from the 
                                   left by 1/4 and from the right by -1/4 of 
                                   the width. Result would be
                                   
                                                  0____________1
                                                   |//|    |//|
                                                   |//|____|//|
                                                     .25  .75
        
           canny_edge_detection : If True, operates an edge detection operation on the cropped image
           reshape_func : A function to reshape the output images into a format that can be read by your classifier
           train_size, validation_size, test_size : self explanatory
    """
    train_size, validation_size, test_size = settings["train_size"], settings["validation_size"], settings["test_size"]
    
    if test_size is not None:
        if af.count_n_images(settings["folder"], settings["extension"]) == settings["train_size"] + settings["validation_size"]:
            af.add_test_set_tofolder(settings["test_folder"], settings["folder"], 
                                     settings["train_size"]+settings["validation_size"], 
                                     settings["extension"])
        
    
    imgmgr = mgr.ImageManager()
    imgmgr.load_images_from_folder(settings["folder"], 
                                   settings["extension"], 
                                   settings["color"], 
                                   verbose)
    imgmgr.bodypart = settings["bodypart"]
    imges = imgmgr.crop_part(settings["size"], 
                             squaring = settings["squaring"], 
                             x_offsets = settings["x_offsets"], 
                             y_offsets = settings["y_offsets"])
    
    if settings["canny_edge_detection"]:
        imges = imgmgr.canny_images()
        
    X = settings["reshape_func"](imges)
    
    # Divide dataset between train, validation, test set. If we select our best performing model based on train and
    # validation results, then a test set is added to the folder in order to check its performance out of sample
    
    X_train = X[: train_size]
    X_val = X[train_size : train_size+validation_size]
    
    if test_size is not None:
        X_test = X[train_size+validation_size : train_size+validation_size+test_size]
        
    else:
        X_test = []
        
    return X_train, X_val, X_test
    
def load_labels(settings):
    """Loads the labels csv file and returns a label vector based on the settings of the
       task. The label vector is split between train, validation and test sets."""
    USECOLS = settings["labels_file_columns"]
    train_size, validation_size, test_size = settings["train_size"], settings["validation_size"], settings["test_size"]
    
    if test_size is not None:
        source_file = settings["test_labels_filename"]
        destination_file = settings["labels_filename"]
        labels_file = af.add_test_set_labels_tofolder(source_file, destination_file, 
                                                      settings["train_size"]+settings["validation_size"], 
                                                      settings["extension"], USECOLS, settings["key_labels_col"]) 
    else:
        labels_file = pd.read_csv(settings["labels_filename"], sep="\t", usecols=USECOLS)
    
    labels = labels_file[settings["labelname"]].values
    labels[labels == -1] = 0 # converts the labels of females and not smiling images to zero
    
    
    
    y_train = labels[: train_size]
    y_val = labels[train_size : train_size+validation_size]
    
    if test_size is not None:
        y_test = labels[train_size+validation_size : train_size+validation_size+test_size]
    else:
        y_test = []
    
    return y_train, y_val, y_test

f = lambda z, x: preprocessing(z.settings[x], False)
data_preprocessing_A1 = lambda: f(cfga, "A1")
data_preprocessing_A2 = lambda: f(cfga, "A2")
data_preprocessing_B1 = lambda: f(cfgb, "B1")
data_preprocessing_B2 = lambda: f(cfgb, "B2")

g = lambda z, x: load_labels(z.settings[x])
labels_A1 = lambda: g(cfga, "A1")
labels_A2 = lambda: g(cfga, "A2")
labels_B1 = lambda: g(cfgb, "B1")
labels_B2 = lambda: g(cfgb, "B2")
    
    
    
    
    
    
    