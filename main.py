import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers



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

# test_labels = np.load('data_sets/y_test.npy',
#                        mmap_mode = None,
#                        allow_pickle = False,
#                        #fix_imports = True,
#                        encoding = 'ASCII')

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

model = keras.Sequential ([ keras.layers.Flatten(input_shape = (300, 2)),  #(None, 300, 2)
                             keras.layers.Dense(8, activation = "relu"),
                             keras.layers.Dense(16, activation = "selu"),
                             keras.layers.Dense(8, activation = "relu"),
                             keras.layers.Dense(4, activation = "relu"),
                             # keras.layers.Dense(4, activation = "selu"),
                             # keras.layers.Dense(4, activation = "relu"),
                             keras.layers.Dense(5, activation = "softmax") ])

# model = keras.Sequential()
# model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,)))
# model.add(layers.Activation('softmax'))

model.compile(  loss = keras.losses.SparseCategoricalCrossentropy(),
                #optimizer = keras.optimizers.Adam(),
                metrics = "accuracy" )



optimalizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile (loss = 'categorical_crossentropy', optimizer = optimalizer, metrics=['Accuracy'])



model.fit (train_data, train_labels, epochs=6)

# test_loss, test_acc = model.evaluate(train_data,  train_labels, verbose=0)
#
# print("test loss: ", test_loss)
# print("test acc: ", test_acc)

print(model.metrics)
#results = model.evaluate(val_data, val_labels, batch_size=64)
# print("test loss, test acc:", results)
#print(results)


# results = model.evaluate(val_data, val_labels, batch_size=128)
# #
# print("test loss, test acc:", results[0], results[1])

# score = model.evaluate(val_data, val_labels, verbose=0)
# print('Test loss:', score[0])
# print('\n')
# print('Test accuracy:', score[1])


#