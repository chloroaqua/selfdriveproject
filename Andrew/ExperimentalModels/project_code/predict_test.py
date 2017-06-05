
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


seq_frames = 5
num_seqs = 5
batch_size = 128

model = load_model('../models/project_model_3dconv_lstm_best.h5', custom_objects={'rmse': util.rmse})
x, y = next(util.generate_arrays_from_file_new_3d_seq(validation_labels, validation_index_center, image_base_path_validation, batch_size, scale=1, number_of_frames=seq_frames, seq_length=num_seqs))
y_pred = model.predict(x)
print(y_pred)
print(y)
print(np.sqrt(np.sum((y-y_pred)**2)/(seq_frames*batch_size)))
