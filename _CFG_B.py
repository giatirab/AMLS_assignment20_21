# Import libraries
import numpy as np
import cv2

parent_folder = "/Users/macbookpro/UCL - MSc Integrated Machine Learning Systems/Y1/Applied Machine Learning I/Final Assignment/AMLS_20-21_SN17024244/Datasets/cartoon_set/"
parent_test_folder = "/Users/macbookpro/UCL - MSc Integrated Machine Learning Systems/Y1/Applied Machine Learning I/Final Assignment/AMLS_20-21_SN17024244/Datasets/cartoon_set_test/"
imgdir = parent_folder + "img/"
imgdir_test = parent_test_folder + "img/"
extension = ".png"

settings = {"B1" :           {"size" : (16, 16),
                              "squaring" : False,
                              "bodypart" : "eyes",
                              "x_offsets" : (1/4,-1/4),
                              "y_offsets" : (1/4,-1/4),
                              "reshape_func" : lambda X : X.reshape(X.shape[0], np.prod(X.shape[1:])),
                              "train_size" : 7500,
                              "validation_size" : 2500,
                              "test_size" : 2500,
                              "folder" : imgdir,
                              "test_folder": imgdir_test,
                              "extension" : extension,
                              "color" : cv2.COLOR_BGR2RGB,
                              "canny_edge_detection" : False,
                              "labels_filename" : parent_folder + "labels.csv",
                              "test_labels_filename": parent_test_folder + "labels.csv",
                              "labels_file_columns" : ["eye_color", "face_shape", "file_name"],
                              "key_labels_col": "file_name",
                              "labelname" : "eye_color"},
            
            "B2" :           {"size" : (100, 100),
                              "squaring" : False,
                              "bodypart" : "face",
                              "x_offsets" : (0,0),
                              "y_offsets" : (0,0),
                              "reshape_func" : lambda X : X.reshape(X.shape[0], np.prod(X.shape[1:])),
                              "train_size" : 7500,
                              "validation_size" : 2500,
                              "test_size" : 2500,
                              "folder" : imgdir,
                              "test_folder": imgdir_test,
                              "extension" : extension,
                              "color" : cv2.COLOR_BGR2RGB,
                              "canny_edge_detection" : True,
                              "labels_filename" : parent_folder + "labels.csv",
                              "test_labels_filename": parent_test_folder + "labels.csv",
                              "labels_file_columns" : ["eye_color", "face_shape", "file_name"],
                              "key_labels_col": "file_name",
                              "labelname" : "face_shape"}
            }
              