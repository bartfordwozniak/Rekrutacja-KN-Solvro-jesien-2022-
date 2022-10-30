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
import csv

# from livelossplot import PlotLossesKeras

#-------------------------------------------------------------POBIERANIE DANYCH--------------------------------------------------

#diffusion_types_names = ['attm', 'crtw', 'fbm', 'lw', 'sbm']

train_data = np.load ('data_sets/X_train.npy',
                       mmap_mode = None,
                       allow_pickle = False,
                       fix_imports = True,
                       encoding = 'ASCII')

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

model = keras.Sequential ([ keras.layers.Flatten(input_shape = (None, 300, 2)),
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

model.summary()



#-------------------------------------------------------TRENING-------------------------------------------------

model.fit(train_data,
          train_labels,
          epochs = 3,
          batch_size = 16,
          )


print(model.metrics)

# loss, acc = model.evaluate(val_data, val_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

#model.load_weights(checkpoint_path)

# loss, acc = model.evaluate(val_data, val_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#!mkdir -p saved_model
#model.save('saved_model/my_model')



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

#dataframe = pd.write_csv(csv_file)

file = open('score.csv', "w")
# header = ['index', 'class']
# writer.writerow(header)
number_of_records = len(test_data)
for i in range (0, number_of_records):
    index = i
    prediction = model.fit(test_data[index])   # fit ????? jakie coś tu musi być?
    #file.writerow([index, prediction])
    file.write([index, prediction])

# classification = pd.DataFrame([ ], columns = ['index', 'class'])
# classification.to_csv('score.csv')

file.close()