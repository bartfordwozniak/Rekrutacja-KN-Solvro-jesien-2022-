#
# ZADANIE REKRUTACYJNE DO KN SOLVRO
#     Politechnika Wrocławska
#       sem. zimowy 22/23
#          Bartosz W.
#           W12N, IMM
#
#--------------------------------------------------------------IMPORTY-------------------------------------------------
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd

# from livelossplot import PlotLossesKeras

#-------------------------------------------------------------POBIERANIE DANYCH--------------------------------------------------

#diffusion_types_names = ['attm', 'crtw', 'fbm', 'lw', 'sbm']

train_data = np.load ('data_sets/X_train.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True)

train_labels = np.load ('data_sets/y_train.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')

test_data = np.load ('data_sets/X_test.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')

# test_labels = np.load('data_sets/y_test.npy',
#                        mmap_mode = None,
#                        allow_pickle = False,
#                        #fix_imports = True,
#                        encoding = 'ASCII')

val_data = np.load ('data_sets/X_val.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')

val_labels = np.load ('data_sets/y_val.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')

#----------------------------------------------------ARCHITEKTURA-----------------------------------------------------

model = keras.Sequential ([ keras.layers.Flatten(input_shape = (300, 2)),  #(None, 300, 2)
                            keras.layers.Dense(128, activation = "relu"),
                            keras.layers.Dense(256, activation = "selu"),
                            keras.layers.Dense(256, activation = "sigmoid"),
                            keras.layers.Dense(128, activation = "relu"),
                            keras.layers.Dense(5, activation = "softmax") ])

#optimalizer = keras.optimizers.Adam(learning_rate=0.01)
optimalizer = keras.optimizers.SGD(learning_rate=0.01)

model.compile (loss = 'categorical_crossentropy',
               optimizer = optimalizer,
               metrics=['Accuracy'])

#-------------------------------------------------------TRENING-------------------------------------------------

model.fit(train_data,
          train_labels,
          epochs = 3,
          batch_size = 4,
          #validation_data=(),
          verbose=1,
          shuffle=True,
          #callbacks=callbacks_list
          )


print(model.metrics)



#-----------------------------------------PRÓBY GENEROWANIA WYKRESÓW---------------------------------------------

# plt.plot(history.history['Accuracy'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# OR (oba coś nie działają, overstack be like xd)

# loss_train = history.history['train_loss']
# loss_val = history.history['val_loss']
# epochs = range(1,35)
# plt.plot(epochs, loss_train, 'g', label='Training loss')
# plt.plot(epochs, loss_val, 'b', label='validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

#-------------------------------------------------------KLASYFIKACJA----------------------------

dataframe = pd.write_csv(csv_file)
col_names = ['index',
             'class',]
