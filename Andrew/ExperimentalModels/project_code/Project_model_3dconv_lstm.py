import os

import keras
import numpy as np
from keras import Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv2D, Flatten, Dense, Conv3D, BatchNormalization, LSTM, TimeDistributed, MaxPooling3D, \
    AveragePooling3D
from keras.models import Sequential, load_model
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
num_seqs = 5


xin = Input(shape=(num_seqs, seq_frames, 120, 320, 3))
c = TimeDistributed(Conv3D(3, (3, 3, 3), strides=(1, 3, 3), activation='relu', use_bias=True, padding='SAME'))(xin)
print(c._keras_shape, "con3d1")
cs = BatchNormalization(axis=4)(c)
print(cs._keras_shape, "bn1")

cs = TimeDistributed(MaxPooling3D(pool_size=(1, 2, 2)))(cs)
cs = TimeDistributed(MaxPooling3D(pool_size=(1, 2, 2)))(cs)

for i in range(2):
    c = TimeDistributed(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', use_bias=True, padding='SAME'))(cs)
    bn1 = BatchNormalization(axis=4)(c)
    print(bn1._keras_shape, "bn1a")
    c = TimeDistributed(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', use_bias=True, padding='SAME'))(bn1)
    bn = BatchNormalization(axis=4)(c)
    print(bn._keras_shape, "bn1b")
    cs = keras.layers.add([bn1, bn])
    print(cs._keras_shape, "res_block{}".format(i))

for i in range(3):
    c = TimeDistributed(Conv3D(8, (3, 3, 3), strides=(1, 2, 2), activation='relu', use_bias=True, padding='SAME'))(cs)
    cs = BatchNormalization(axis=4)(c)
    print(cs._keras_shape, "shrink_block{}".format(i))


#cs = TimeDistributed(AveragePooling3D(pool_size=(1, 2, 2)))(cs)
#cs = TimeDistributed(MaxPooling3D(pool_size=(1, 2, 2)))(cs)


flt = TimeDistributed(Flatten())(cs)
print(flt._keras_shape)
lstm1 = LSTM(64, activation='tanh', return_sequences=True, implementation=2)(flt)
print(lstm1._keras_shape, "lstm1")
lstm2 = LSTM(16,  activation='tanh', return_sequences=True, implementation=2)(lstm1)
print(lstm2._keras_shape, "lstm2")
d = TimeDistributed(Dense(512))(lstm2)
print(d._keras_shape, "dense1")
d = TimeDistributed(Dense(128, activation='relu'))(d)
d = TimeDistributed(Dense(64, activation='relu'))(d)
d = TimeDistributed(Dense(16, activation='relu'))(d)
angle = TimeDistributed(Dense(1))(d)
model = Model(inputs=xin, outputs=angle)

learning_rate = 0.001
decay_rate = learning_rate / 32
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)

model.load_weights("project_model_3dconv_lstm_check.hdf5")
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[util.rmse])

learning_rate = 0.001
decay_rate = learning_rate / 32
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)

model.summary()

history = util.LossHistory()
lrate = LearningRateScheduler(util.step_decay)
checkpointer = ModelCheckpoint(filepath="project_model_3dconv_lstm_check_new.hdf5", verbose=1, save_best_only=True)

print(util.std_evaluate(model, util.generate_arrays_from_file_new_3d_seq(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1, number_of_frames=seq_frames, seq_length=num_seqs), 64))

model.fit_generator(util.generate_arrays_from_file_new_3d_seq(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1, number_of_frames=seq_frames, seq_length=num_seqs),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=util.generate_arrays_from_file_new_3d_seq(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1, number_of_frames=seq_frames, seq_length=num_seqs),
                    validation_steps=validation_labels.shape[0] // 32, epochs=2, verbose=1, callbacks=[history, checkpointer, lrate])

model.save('../models/project_model_3dconv_lstm_best.h5')
print(util.std_evaluate(model, util.generate_arrays_from_file_new_3d_seq(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1, number_of_frames=seq_frames, seq_length=num_seqs), 32))
