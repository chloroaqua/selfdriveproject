import os

import keras
import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.engine import Model
from keras.layers import Conv2D, Flatten, Dense, Conv3D, BatchNormalization, LSTM, TimeDistributed, MaxPooling2D, \
    GlobalAveragePooling2D, Reshape
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.optimizers import Adam

import project_code.util as util
import cv2

from project_code.Pretrain_Hack.resnet50 import ResNet50

training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out3\\training"
training_labels_center = np.load(os.path.join(training_dataset_path, 'training_center_labels.npy'))
training_index_center = np.load(os.path.join(training_dataset_path, 'training_center_indexes.npy'))
image_base_path_training_center = os.path.join(training_dataset_path, 'images\\center')

validation_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out3\\validation"
validation_labels = np.load(os.path.join(validation_dataset_path, 'validation_center_labels.npy'))
validation_index_center = np.load(os.path.join(validation_dataset_path, 'validation_center_indexes.npy'))
image_base_path_validation = os.path.join(validation_dataset_path, 'images\\center')


#Check to see what would happen if we predicted all 0s
rmse_test = np.sqrt(np.mean(np.square(training_labels_center)))
print(rmse_test)
rmse_test = np.sqrt(np.mean(np.square(validation_labels)))
print(rmse_test)


seq_len = 5
#num_seqs = 4
input_layer = Input(shape=(seq_len, 224, 224, 3))
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(seq_len, 224, 224, 3))
print(len(resnet.layers))
for layer in resnet.layers[:160]:
   layer.trainable = False
for layer in resnet.layers[160:]:
   layer.trainable = True

resnet = resnet(input_layer)

flt = TimeDistributed(Flatten())(resnet)
print(flt._keras_shape)
lstm1 = LSTM(64, activation='tanh', return_sequences=True)(flt)
print(lstm1._keras_shape, "lstm1")
lstm2 = LSTM(16,  activation='tanh', return_sequences=True)(lstm1)
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

learning_rate = 0.001
decay_rate = learning_rate / 32
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)

model = Model(inputs=input_layer, outputs=angle)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[util.rmse])

model.summary()

history = util.LossHistory()
lrate = LearningRateScheduler(util.step_decay)
checkpointer = ModelCheckpoint(filepath="res50_trans_net_lstm_test_check.hdf5", verbose=1, save_best_only=True)
model.fit_generator(util.generate_arrays_from_file_new_3d(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1, input_shape=(224, 224, 3), random_flip=True, number_of_frames=seq_len),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=util.generate_arrays_from_file_new_3d(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1, input_shape=(224, 224, 3)),
                    validation_steps=validation_labels.shape[0] // 32, epochs=32, verbose=1, callbacks=[history, checkpointer, lrate])


model.save('../models/res50_trans_net_lstm_test.h5')
print(history.losses)



