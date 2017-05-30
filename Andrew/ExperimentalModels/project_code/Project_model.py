import os

import numpy as np
from keras.layers import Conv2D, Flatten, Dense, Conv3D, BatchNormalization, LSTM, TimeDistributed
from keras.models import Sequential
import matplotlib.pyplot as plt
import project_code.util as util
import cv2

training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
training_labels_center = np.load(os.path.join(training_dataset_path, 'training_center_labels.npy'))
training_index_center = np.load(os.path.join(training_dataset_path, 'training_center_indexes.npy'))
image_base_path_training_center = os.path.join(training_dataset_path, 'images\\center')

validation_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation"
validation_labels = np.load(os.path.join(validation_dataset_path, 'validation_center_labels.npy'))
validation_index_center = np.load(os.path.join(validation_dataset_path, 'validation_center_indexes.npy'))
image_base_path_validation = os.path.join(validation_dataset_path, 'images\\center')


#Check to see what would happen if we predicted all 0s
rmse_test = np.sqrt(np.mean(np.square(training_labels_center)))
print(rmse_test)
rmse_test = np.sqrt(np.mean(np.square(validation_labels)))
print(rmse_test)


num_frames = 10
model = Sequential()
model.add(Conv3D(24, (3, 3, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME', input_shape=(num_frames, 120, 320, 3)))
print(model.layers[-1].output_shape)
model.add(BatchNormalization())
model.add(Conv3D(64, (3, 3, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME', input_shape=(num_frames, 120, 320, 3)))
print(model.layers[-1].output_shape)
model.add(BatchNormalization())
model.add(Conv3D(128, (3, 3, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME', input_shape=(num_frames, 120, 320, 3)))
print(model.layers[-1].output_shape)
model.add(BatchNormalization())
model.add(TimeDistributed(Flatten()))
print(model.layers[-1].output_shape)
model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2))
print(model.layers[-1].output_shape)
model.add(Dense(10, activation='relu'))
print(model.layers[-1].output_shape)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[util.rmse])



model.fit_generator(util.generate_arrays_from_file_new_3d(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1, number_of_frames=num_frames),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=util.generate_arrays_from_file_new_3d(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1, number_of_frames=num_frames),
                    validation_steps=validation_labels.shape[0] // 32, epochs=2, verbose=1)

model.save('../models/3d_test_v2.h5')