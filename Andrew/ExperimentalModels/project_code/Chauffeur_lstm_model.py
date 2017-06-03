import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Flatten, Dense, Dropout, SpatialDropout2D, MaxPooling2D, TimeDistributed, LSTM
from keras.models import Sequential
from keras.regularizers import l2

import project_code.util as util


training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
training_labels_center = np.load(os.path.join(training_dataset_path, 'training_center_labels.npy'))
training_index_center = (np.load(os.path.join(training_dataset_path, 'training_center_indexes.npy')).astype(int))
image_base_path_training_center = os.path.join(training_dataset_path, 'images\\center')

validation_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation"
validation_labels = np.load(os.path.join(validation_dataset_path, 'validation_center_labels.npy'))
validation_index_center = (np.load(os.path.join(validation_dataset_path, 'validation_center_indexes.npy')).astype(int))
image_base_path_validation = os.path.join(validation_dataset_path, 'images\\center')

rmse_test = np.sqrt(np.mean(np.square(training_labels_center)))
print(rmse_test, "loaded labels rmse")

num_frames = 50
model = Sequential()
model.add(TimeDistributed(Conv2D(24, 5, 5,
                                        init="he_normal",
                                        activation='relu',
                                        subsample=(5, 4),
                                        border_mode='valid'), input_shape=(num_frames, 120, 320, 3)))
model.add(TimeDistributed(Conv2D(32, 5, 5,
                                        init="he_normal",
                                        activation='relu',
                                        subsample=(3, 2),
                                        border_mode='valid')))
model.add(TimeDistributed(Conv2D(48, 3, 3,
                                        init="he_normal",
                                        activation='relu',
                                        subsample=(1, 2),
                                        border_mode='valid')))
model.add(TimeDistributed(Conv2D(64, 3, 3,
                                        init="he_normal",
                                        activation='relu',
                                        border_mode='valid')))
model.add(TimeDistributed(Conv2D(128, 3, 3,
                                        init="he_normal",
                                        activation='relu',
                                        subsample=(1, 2),
                                        border_mode='valid')))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True, implementation=2))
model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True, implementation=2))
model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, implementation=2))
model.add(Dropout(0.2))
model.add(Dense(
    output_dim=256,
    init='he_normal',
    activation='relu',
    W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(
    output_dim=1,
    init='he_normal',
    W_regularizer=l2(0.001)))

model.load_weights("../models/tmp/chauffeur_lstm_check.hdf5")

model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=[util.rmse])

#Check a batch to see the base of the model
#print(util.std_evaluate(model, util.generate_arrays_from_file_new_3d(validation_labels, validation_index_center, image_base_path_validation, 32, number_of_frames=num_frames), validation_index_center.shape[0]//32))
history = util.LossHistory()
checkpointer = ModelCheckpoint(filepath="../models/tmp/chauffeur_lstm_check.hdf5", verbose=1, save_best_only=True)
model.summary()
#Train the model with the generators as the dataset is too large to keep in memory
model.fit_generator(util.generate_arrays_from_file_new_3d(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1, number_of_frames=num_frames, random_flip=True),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=util.generate_arrays_from_file_new_3d(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1, number_of_frames=num_frames),
                    validation_steps=validation_labels.shape[0] // 32, epochs=40, verbose=1, callbacks=[history, checkpointer])


model.save('../models/chauffeur_lstm_model_40_epoch.h5')

#check a batch on the validation set after training
print(util.std_evaluate(model, util.generate_arrays_from_file_new_3d(validation_labels, validation_index_center, image_base_path_validation, 32, number_of_frames=num_frames), validation_index_center.shape[0]//32))
