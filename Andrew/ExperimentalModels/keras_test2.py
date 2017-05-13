import numpy as np
import tensorflow as tf
import keras
import os
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error

import keras_help

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization

training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
training_labels = np.load(os.path.join(training_dataset_path, 'training_center_labels.npy'))
training_index_center = np.load(os.path.join(training_dataset_path, 'training_center_indexes.npy'))
image_base_path_training = os.path.join(training_dataset_path, 'images\\center')


validation_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation"
validation_labels = np.load(os.path.join(validation_dataset_path, 'validation_center_labels.npy'))
validation_index_center = np.load(os.path.join(validation_dataset_path, 'validation_center_indexes.npy'))
image_base_path_validation = os.path.join(validation_dataset_path, 'images\\center')


'''
for val in keras_help.generate_arrays_from_file_v2(training_labels, training_index_center, image_base_path_training, 2):
    image, y = val
    print(image)
    print(y)
    print(image.shape)
    #plt.imshow(image)
'''

print(training_labels.shape)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(240, 320, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras_help.rmse])
'''
model.fit_generator(keras_help.generate_arrays_from_file_v2(training_labels, training_index_center, image_base_path_training, 32),
                    steps_per_epoch=training_labels.shape[0] // 32, epochs=10, verbose=1)
'''
model.fit_generator(keras_help.generate_arrays_from_file_v2(training_labels, training_index_center, image_base_path_training, 32),
                    steps_per_epoch=training_labels.shape[0] // 32,
                    validation_data=keras_help.generate_arrays_from_file_v2(validation_labels, validation_index_center, image_base_path_validation, 32),
                    validation_steps=validation_labels.shape[0] // 32, epochs=40, verbose=1)






