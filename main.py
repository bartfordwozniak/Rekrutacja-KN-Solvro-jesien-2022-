# This is a sample Python script.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn
import os
import jax, jaxlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#print(tf.__version__)

diffusion_types_names = ['attm', 'crtw', 'fbm', 'lw', 'sbm']       #class names

x_train_data = np.load('data_sets/X_train.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')

y_train_data = np.load('data_sets/y_train.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')
                        #encoding = 'OneHot')

x_test_data = np.load('data_sets/X_test.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')

x_val_data = np.load('data_sets/X_val.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')

y_val_data = np.load('data_sets/y_val.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')
                        #encoding = 'OneHot')


#x_train_data = np.array()
#print(x_train_data)
print(x_train_data.shape)
print(y_train_data.shape)
print(x_test_data.shape)


# Divide the dataset into 3 even parts, each containing 1/3 of the data
#x_train_data_split_0, x_train_data_split_1, x_train_data_split_2 = tfds.even_splits('train', n=3)
#ds = tfds.load('my_dataset', split=split2)

#split = tfds.split_for_jax_process('train', drop_remainder=True)
#ds = tfds.load(x_train_data, split=split)





################################################### skrypt sprawdzający czy nie brakuje danych

#create the model
model = tf.keras.Sequential ([ tf.keras.layers.Flatten(input_shape=(49000, 300, 2)),             #czy jest potrzeba spłaszczania danych??????
                             tf.keras.layers.Dense(4, activation="relu"),
                             tf.keras.layers.Dense(4, activation="relu"),
                             #tf.keras.layers.Dense(4, activation="relu"),
                             tf.keras.layers.Dense(5, activation="softmax") ])


# model.compile(  loss = tf.keras.losses.SparseCategoricalCrossentropy(),
#                 optimizer = tf.keras.optimizers.Adam(),
#                 metrics = "accuracy" )

# #create a learning rate callback
# lr_scheduler = tf.keras.callbacks
#     .LearningRateScheduler(lambda epoch : 1e-3 *10**(epoch/20) )
#
# #fit the model
# fit_lr_history = model.fit(
#    train_data_norm,
#    train_labels,
#    epochs=40,
#    callbacks=[lr_scheduler],
#    validation_data=(test_data_norm,test_labels))









# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline

# load dataset
# dataframe = pandas.read_csv("iris.data", header=None)
# dataset = dataframe.values
# X = dataset[:, 0:4].astype(float)
# Y = dataset[:, 4]
# # encode class values as integers
#
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = np_utils.to_categorical(encoded_Y)
#
#
# # define baseline model
# def baseline_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(8, input_dim=4, activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
#
# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


