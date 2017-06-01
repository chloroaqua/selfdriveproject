import os

import numpy as np
from keras import Input
from keras.engine import Model
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


seq_frames = 5
num_seqs = 4


xin = Input(batch_shape=(32, num_seqs, seq_frames, 120, 320, 3))
con3d1 = TimeDistributed(Conv3D(64, (12, 12, 3), strides=(1, 6, 6), activation='relu', use_bias=True, padding='SAME'))(xin)
print(con3d1._keras_shape, "con3d1")
bn1 = BatchNormalization(axis=4)(con3d1)
print(bn1._keras_shape, "bn1")

con3d2 = TimeDistributed(Conv3D(64, (5, 5, 2), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME'))(bn1)
bn2 = BatchNormalization(axis=4)(con3d2)
print(bn2._keras_shape, "bn2")

con3d3 = TimeDistributed(Conv3D(64, (5, 5, 2), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME'))(bn2)
bn3 = BatchNormalization(axis=4)(con3d3)
print(bn3._keras_shape, "bn3")

con3d4 = TimeDistributed(Conv3D(64, (5, 5, 2), strides=(2, 2, 1), activation='relu', use_bias=True, padding='SAME'))(bn3)
bn4 = BatchNormalization(axis=4)(con3d4)
print(bn4._keras_shape, "bn4")


flt = TimeDistributed(Flatten())(bn4)
print(flt._keras_shape)
lstm1 = LSTM(64, activation='tanh', return_sequences=True)(flt)
print(lstm1._keras_shape, "lstm1")
lstm2 = LSTM(16,  activation='tanh', return_sequences=True)(lstm1)
print(lstm2._keras_shape, "lstm2")
dense1 = TimeDistributed(Dense(10))(lstm2)
print(dense1._keras_shape, "dense1")
angle = TimeDistributed(Dense(1))(dense1)
model = Model(inputs=xin, outputs=angle)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[util.rmse])

# model = Sequential()
# model.add(TimeDistributed(Conv3D(24, (3, 3, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME'), input_shape=(num_seqs, seq_frames, 120, 320, 3)))
# #model.add(TimeDistributed(Conv2D(24, (5, 5), strides=2, activation='relu'), input_shape=(num_frames, 120, 320, 3)))
# print(model.layers[-1].output_shape)
# model.add(BatchNormalization())
# model.add(TimeDistributed(Conv3D(3, (5, 5, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME')))
# #model.add(TimeDistributed(Conv2D(24, (5, 5), strides=2, activation='relu')))
# print(model.layers[-1].output_shape)
# model.add(BatchNormalization())
# #model.add(Conv3D(128, (3, 3, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME', input_shape=(num_frames, 120, 320, 3)))
# print(model.layers[-1].output_shape)
# model.add(BatchNormalization())
# model.add(TimeDistributed(Flatten()))
# print(model.layers[-1].output_shape)
# #model.add(LSTM(64, activation='tanh', return_sequences=True))
# #model.add(LSTM(64, activation='tanh', return_sequences=True))
# model.add(LSTM(64,  activation='tanh'))
# print(model.layers[-1].output_shape)
# model.add(TimeDistributed(Dense(8, activation='relu')))
# print(model.layers[-1].output_shape)
# model.add(TimeDistributed(Dense(1)))
#
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=[util.rmse])
#


model.fit_generator(util.generate_arrays_from_file_new_3d_seq(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1, number_of_frames=seq_frames, seq_length=num_seqs),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=util.generate_arrays_from_file_new_3d_seq(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1, number_of_frames=seq_frames, seq_length=num_seqs),
                    validation_steps=validation_labels.shape[0] // 32, epochs=32, verbose=1)

model.save('../models/3d_test_seq_v1.h5')