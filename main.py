# This is a sample Python script.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn
import jax, jaxlib


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #naprawa pewnego błędu, zapożyczone z stackoverflow

#print(tf.__version__)

#diffusion_types_names = ['attm', 'crtw', 'fbm', 'lw', 'sbm']

train_data = np.load('data_sets/X_train.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')

train_labels = np.load('data_sets/y_train.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')
                        #encoding = 'OneHot')

test_data = np.load('data_sets/X_test.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')

val_data = np.load('data_sets/X_val.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')

val_labels = np.load('data_sets/y_val.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       #fix_imports = True,
                       encoding = 'ASCII')
                        #encoding = 'OneHot')

model = tf.keras.Sequential ([ tf.keras.layers.Flatten(input_shape=(None, 300, 2)),
                             tf.keras.layers.Dense(4, activation="relu"),
                             tf.keras.layers.Dense(4, activation="relu"),
                             #tf.keras.layers.Dense(4, activation="relu"),
                             tf.keras.layers.Dense(5, activation="softmax") ])


model.compile(  loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(),
                metrics = "accuracy" )

model.fit (train_data, train_labels, epochs=5)


#
