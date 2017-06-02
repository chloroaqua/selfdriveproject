import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Flatten, Dense, Dropout, SpatialDropout2D, MaxPooling2D
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

model = Sequential()
model.add(Conv2D(16, 5, 5,
        input_shape=(120, 320, 3),
        init= "he_normal",
        activation='relu',
        border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, 5, 5,
        init= "he_normal",
        activation='relu',
        border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(40, 3, 3,
        init= "he_normal",
        activation='relu',
        border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(60, 3, 3,
        init= "he_normal",
        activation='relu',
        border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(80, 2, 2,
        init= "he_normal",
        activation='relu',
        border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, 2, 2,
        init= "he_normal",
        activation='relu',
        border_mode='same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(
        output_dim=1,
        init='he_normal',
        W_regularizer=l2(0.0001)))
model.compile(loss='mean_squared_error', optimizer='adadelta',  metrics=[util.rmse])

#Check a batch to see the base of the model
print(util.std_evaluate(model, util.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), validation_index_center.shape[0]//32))
history = util.LossHistory()
checkpointer = ModelCheckpoint(filepath="../models/tmp/chauffeur_check.hdf5", verbose=1, save_best_only=True)
model.summary()
#Train the model with the generators as the dataset is too large to keep in memory
model.fit_generator(util.generate_arrays_from_file_new(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=util.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1),
                    validation_steps=validation_labels.shape[0] // 32, epochs=100, verbose=1, callbacks=[history, checkpointer])


model.save('../models/chauffeur_regression_model_100_epoch.h5')

#check a batch on the validation set after training
print(util.std_evaluate(model, util.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), validation_index_center.shape[0]//32))

