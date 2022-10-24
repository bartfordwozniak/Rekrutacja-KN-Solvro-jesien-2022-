import numpy as np
from tensorflow import keras
import tensorflow as tf
#print(tf.__version__)

#diffusion_types_names = ['attm', 'crtw', 'fbm', 'lw', 'sbm']

train_data = np.load('data_sets/X_train.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')

train_labels = np.load('data_sets/y_train.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')
                        #encoding = 'OneHot')

test_data = np.load('data_sets/X_test.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')

# test_labels = np.load('data_sets/y_test.npy',
#                        mmap_mode = None,
#                        allow_pickle = False,
#                        #fix_imports = True,
#                        encoding = 'ASCII')

val_data = np.load('data_sets/X_val.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')

val_labels = np.load('data_sets/y_val.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')
                        #encoding = 'OneHot')

model = keras.Sequential ([ keras.layers.Flatten(input_shape = (300, 2)),  #(None, 300, 2)
                            keras.layers.Dense(128, activation = "relu"),
                            keras.layers.Dense(256, activation = "selu"),
                            keras.layers.Dense(128, activation = "sigmoid"),
                            keras.layers.Dense(5, activation = "softmax") ])

#optimalizer = keras.optimizers.Adam(learning_rate=0.01)
optimalizer = keras.optimizers.SGD(learning_rate=0.005)

model.compile (loss = 'categorical_crossentropy', optimizer = optimalizer, metrics=['Accuracy'])

# val_set = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
# val_set = val_set.batch(64)

model.fit (train_data, train_labels, batch_size=4, epochs=8) #,validation_data = val_set)

print(model.metrics)
