## ======================================================================================================================
# Import libraries

from sklearn.pipeline import make_pipeline
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
import tensorflow as tf
import _LIB_DataProcessing as dp
import sklearn.metrics
from termcolor import colored
import numpy as np

## ======================================================================================================================
# Data pre-processing

# A1

X_train_A1, X_val_A1, X_test_A1 = dp.data_preprocessing_A1()
Y_train_A1, Y_val_A1, Y_test_A1 = dp.labels_A1()

# A2

X_train_A2, X_val_A2, X_test_A2 = dp.data_preprocessing_A2()
Y_train_A2, Y_val_A2, Y_test_A2 = dp.labels_A2()

# B1

X_train_B1, X_val_B1, X_test_B1 = dp.data_preprocessing_B1()
Y_train_B1, Y_val_B1, Y_test_B1 = dp.labels_B1()

# B2

X_train_B2, X_val_B2, X_test_B2 = dp.data_preprocessing_B2()
Y_train_B2, Y_val_B2, Y_test_B2 = dp.labels_B2()

## ======================================================================================================================
# Task A1 solution - Pipeline on SVM

SVMBase = lm.SGDClassifier(loss='hinge', penalty='l1')

clf_A1 = make_pipeline(MinMaxScaler(), SVMBase)

clf_A1.fit(X_train_A1, Y_train_A1)
y_pred_train_A1 = clf_A1.predict(X_train_A1)
y_pred_val_A1 = clf_A1.predict(X_val_A1)
print(colored("TASK A1 - TRAIN SET METRICS", "cyan"))
print(sklearn.metrics.classification_report(Y_train_A1, y_pred_train_A1))
print(colored("TASK A1 - VALIDATION SET METRICS", "cyan"))
print(sklearn.metrics.classification_report(Y_val_A1, y_pred_val_A1))

if X_test_A1 is not None:
    y_pred_test_A1 = clf_A1.predict(X_test_A1)
    print(colored("TASK A1 - TEST SET METRICS", "cyan"))
    print(sklearn.metrics.classification_report(Y_test_A1, y_pred_test_A1))

## ======================================================================================================================
# Task A2 solution - CNN
     
nb_filters = 16
nb_pool = 2
nb_conv = 3
nb_classes = 2
activation = "relu"

clf_A2 = models.Sequential()

clf_A2.add(layers.Conv2D(nb_filters, (nb_conv, nb_conv), activation=activation, input_shape=X_train_A2.shape[1:]))
clf_A2.add(layers.Conv2D(nb_filters, (nb_conv, nb_conv), activation=activation))
clf_A2.add(layers.MaxPooling2D(pool_size=(nb_pool, nb_pool)))
clf_A2.add(layers.Dropout(0.25))
clf_A2.add(layers.Flatten())
clf_A2.add(layers.Dense(64, activation=activation))
clf_A2.add(layers.Dropout(0.5))
clf_A2.add(layers.Dense(nb_classes))

clf_A2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = clf_A2.fit(X_train_A2, Y_train_A2, epochs=10, validation_data=(X_val_A2, Y_val_A2), verbose=False)

y_pred_train_A2 = np.argmax(clf_A2.predict(X_train_A2), axis=1)
y_pred_val_A2 = np.argmax(clf_A2.predict(X_val_A2), axis=1)
print(colored("TASK A2 - TRAIN SET METRICS", "cyan"))
print(sklearn.metrics.classification_report(Y_train_A2, y_pred_train_A2))
print(colored("TASK A2 - VALIDATION SET METRICS", "cyan"))
print(sklearn.metrics.classification_report(Y_val_A2, y_pred_val_A2))

if X_test_A2 is not None:
    y_pred_test_A2 = np.argmax(clf_A2.predict(X_test_A2), axis=1)
    print(colored("TASK A2 - TEST SET METRICS", "cyan"))
    print(sklearn.metrics.classification_report(Y_test_A2, y_pred_test_A2))

## ======================================================================================================================
# Task B1 solution - KNN
    
clf_B1 = KNeighborsClassifier(10, p=2)

clf_B1.fit(X_train_B1, Y_train_B1)
y_pred_train_B1 = clf_B1.predict(X_train_B1)
y_pred_val_B1 = clf_B1.predict(X_val_B1)
print(colored("TASK B1 - TRAIN METRICS", "cyan"))
print(sklearn.metrics.classification_report(Y_train_B1, y_pred_train_B1))
print(colored("TASK B1 - VALIDATION METRICS", "cyan"))
print(sklearn.metrics.classification_report(Y_val_B1, y_pred_val_B1))

if X_test_B1 is not None:
    y_pred_test_B1 = clf_B1.predict(X_test_B1)
    print(colored("TASK B1 - TEST SET METRICS", "cyan"))
    print(sklearn.metrics.classification_report(Y_test_B1, y_pred_test_B1))

## ======================================================================================================================
# Task B2 solution - RANDOM FOREST
    
clf_B2 = RandomForestClassifier(n_estimators=80, max_depth=100, criterion="entropy", max_leaf_nodes=130, max_features="auto")

clf_B2.fit(X_train_B2, Y_train_B2)
y_pred_train_B2 = clf_B2.predict(X_train_B2)
y_pred_val_B2 = clf_B2.predict(X_val_B2)
print(colored("TASK B2 - TRAIN METRICS", "cyan"))
print(sklearn.metrics.classification_report(Y_train_B2, y_pred_train_B2))
print(colored("TASK B2 - VALIDATION METRICS", "cyan"))
print(sklearn.metrics.classification_report(Y_val_B2, y_pred_val_B2))
    
if X_test_B2 is not None:
    y_pred_test_B2 = clf_B2.predict(X_test_B2)
    print(colored("TASK B2 - TEST SET METRICS", "cyan"))
    print(sklearn.metrics.classification_report(Y_test_B2, y_pred_test_B2)) 
    