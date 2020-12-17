# Import libraries
import numpy as np
import cv2

parent_folder = "/Users/macbookpro/UCL - MSc Integrated Machine Learning Systems/Y1/Applied Machine Learning I/Final Assignment/AMLS_20-21_SN17024244/Datasets/celeba/"
parent_test_folder = "/Users/macbookpro/UCL - MSc Integrated Machine Learning Systems/Y1/Applied Machine Learning I/Final Assignment/AMLS_20-21_SN17024244/Datasets/celeba_test/"
imgdir = parent_folder + "img/"
imgdir_test = parent_test_folder + "img/"
extension = ".jpg"

settings = {"A1" :           {"size" : (48, 48),
                              "squaring" : False,
                              "bodypart" : "face",
                              "x_offsets" : (1/8,-1/8),
                              "y_offsets" : (1/8,-1/8),
                              "reshape_func" : lambda X : X.reshape(X.shape[0], np.prod(X.shape[1:])),
                              "train_size" : 3750,
                              "validation_size" : 1250,
                              "test_size" : 1000,
                              "folder" : imgdir,
                              "test_folder": imgdir_test,
                              "extension" : extension,
                              "color" : cv2.COLOR_BGR2RGB,
                              "canny_edge_detection" : False,
                              "labels_filename" : parent_folder + "labels.csv",
                              "test_labels_filename": parent_test_folder + "labels.csv",
                              "labels_file_columns" : ["img_name", "gender", "smiling"],
                              "key_labels_col": "img_name",
                              "labelname" : "gender"},
            
            "A2" :           {"size" : (48, 48),
                              "squaring" : False,
                              "bodypart" : "smile",
                              "x_offsets" : (1/5,4/5),
                              "y_offsets" : (2/3,1),
                              "reshape_func" : lambda X : X.reshape(X.shape[0], X.shape[1], X.shape[2], 1),
                              "train_size" : 3750,
                              "validation_size" : 1250,
                              "test_size" : 1000,
                              "folder" : imgdir,
                              "test_folder": imgdir_test,
                              "extension" : extension,
                              "color" : cv2.IMREAD_GRAYSCALE,
                              "canny_edge_detection" : False,
                              "labels_filename" : parent_folder + "labels.csv",
                              "test_labels_filename": parent_test_folder + "labels.csv",
                              "labels_file_columns" : ["img_name", "gender", "smiling"],
                              "key_labels_col": "img_name",
                              "labelname" : "smiling"}
            }
                                
                                                              
        