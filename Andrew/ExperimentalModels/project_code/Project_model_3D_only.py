import os

import keras
import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.engine import Model
from keras.layers import Conv2D, Flatten, Dense, Conv3D, BatchNormalization, LSTM, TimeDistributed, MaxPooling2D, \
    MaxPooling3D
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.optimizers import Adam

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


input_layer = Input(batch_shape=(32, seq_frames, 120, 320, 3))

c = Conv3D(24, (3, 3, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='SAME')(input_layer)
cs = BatchNormalization(axis=4)(c)
print(cs._keras_shape, "3d1")
cs = MaxPooling3D(pool_size=(1, 2, 2))(cs)
cs = MaxPooling3D(pool_size=(1, 2, 2))(cs)

for i in range(4):
    c = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', use_bias=True, padding='SAME')(cs)
    bn1 = BatchNormalization(axis=4)(c)
    c = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', use_bias=True, padding='SAME')(bn1)
    bn = BatchNormalization(axis=4)(c)
    c = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', use_bias=True, padding='SAME')(c)
    bn = BatchNormalization(axis=4)(c)
    cs = keras.layers.add([bn1, bn])
    print(cs._keras_shape, "res_block{}".format(i))

cs = MaxPooling3D(pool_size=(1, 2, 2))(cs)
cs = MaxPooling3D(pool_size=(1, 2, 2))(cs)

# for i in range(1):
#     c = Conv3D(24, (3, 3, 3), strides=(2, 2, 2), activation='relu', use_bias=True, padding='VALID')(cs)
#     cs = BatchNormalization(axis=3)(c)
#     print(cs._keras_shape, "shrink_block{}".format(i))


flt = Flatten()(cs)
dense1 = Dense(512, activation='relu', use_bias=True)(flt)
print(dense1._keras_shape, "dense1")
dense2 = Dense(256, activation='relu', use_bias=True)(dense1)
print(dense2._keras_shape, "dense2")
dense3 = Dense(64, activation='relu', use_bias=True)(dense2)
print(dense3._keras_shape, "dense3")
angle = Dense(1, use_bias=True)(dense3)
print(angle._keras_shape, "angle")


learning_rate = 0.001
decay_rate = learning_rate / 32
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)

model = Model(inputs=input_layer, outputs=angle)
model.load_weights("3d_only_check_check.hdf5")
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[util.rmse])

model.summary()

history = util.LossHistory()
lrate = LearningRateScheduler(util.step_decay)
checkpointer = ModelCheckpoint(filepath="3d_only_check_check.hdf5", verbose=1, save_best_only=True)
model.fit_generator(util.generate_arrays_from_file_new_3d(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1, random_flip=True, number_of_frames=seq_frames),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=util.generate_arrays_from_file_new_3d(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1,  number_of_frames=seq_frames),
                    validation_steps=validation_labels.shape[0] // 32, epochs=32, verbose=1, callbacks=[history, checkpointer, lrate])


model.save('../models/3d_only_test.h5')
print(history.losses)

