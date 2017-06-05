import os

import keras
import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv2D, Flatten, Dense, Conv3D, BatchNormalization, LSTM, TimeDistributed, MaxPooling2D
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
#num_seqs = 4


xin = Input(batch_shape=(32, seq_frames, 120, 320, 3))
con2d1 = TimeDistributed(Conv2D(3, (5, 5), strides=(2, 2), activation='relu', use_bias=True, padding='SAME'))(xin)
print(con2d1._keras_shape, "con3d1")
cs = BatchNormalization(axis=3)(con2d1)
print(cs._keras_shape, "bn1")

cs = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(cs)
print(cs._keras_shape, "max")


for i in range(4):
    c =TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', use_bias=True, padding='SAME'))(cs)
    bn1 = BatchNormalization(axis=3)(c)
    c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', use_bias=True, padding='SAME'))(bn1)
    bn = BatchNormalization(axis=3)(c)
    c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', use_bias=True, padding='SAME'))(c)
    bn = BatchNormalization(axis=3)(c)
    cs = keras.layers.add([bn1, bn])
    print(cs._keras_shape, "res_block{}".format(i))

for i in range(4):
    c = TimeDistributed(Conv2D(64, (1, 1), strides=(2, 2), activation='relu', use_bias=True, padding='VALID'))(cs)
    cs = BatchNormalization(axis=3)(c)
    print(cs._keras_shape, "shrink_block{}".format(i))


flt = TimeDistributed(Flatten())(cs)
print(flt._keras_shape)
lstm1 = LSTM(64, activation='tanh', return_sequences=True, implementation=2)(flt)
print(lstm1._keras_shape, "lstm1")
lstm2 = LSTM(16,  activation='tanh', return_sequences=True, implementation=2)(lstm1)
print(lstm2._keras_shape, "lstm2")
lstm3 = LSTM(16,  activation='tanh')(lstm2)
dense1 = Dense(512, activation='relu', use_bias=True)(lstm3)
print(dense1._keras_shape, "dense1")
dense2 = Dense(256, activation='relu', use_bias=True)(dense1)
print(dense2._keras_shape, "dense2")
dense3 = Dense(64, activation='relu', use_bias=True)(dense2)
print(dense3._keras_shape, "dense3")
angle = Dense(1)(dense3)
print(angle._keras_shape, "angle")
model = Model(inputs=xin, outputs=angle)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[util.rmse])

history = util.LossHistory()
checkpointer = ModelCheckpoint(filepath="../models/tmp/res_net_test_check.hdf5", verbose=1, save_best_only=True)
model.fit_generator(util.generate_arrays_from_file_new_3d(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1, number_of_frames=seq_frames),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=util.generate_arrays_from_file_new_3d(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1, number_of_frames=seq_frames),
                    validation_steps=validation_labels.shape[0] // 32, epochs=32, verbose=1, callbacks=[history, checkpointer])


model.save('../models/res_net_test.h5')
print(history.losses)