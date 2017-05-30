import os

import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, \
    SpatialDropout2D
from keras.models import Sequential
from keras.regularizers import l2

from old import keras_help

training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
training_labels_center = np.load(os.path.join(training_dataset_path, 'training_center_labels.npy'))
training_index_center = np.load(os.path.join(training_dataset_path, 'training_center_indexes.npy'))
image_base_path_training_center = os.path.join(training_dataset_path, 'images\\center')


training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
training_labels_left = np.load(os.path.join(training_dataset_path, 'training_left_labels.npy'))
training_index_left = np.load(os.path.join(training_dataset_path, 'training_left_indexes.npy'))
image_base_path_training_left = os.path.join(training_dataset_path, 'images\\left')


training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
training_labels_right = np.load(os.path.join(training_dataset_path, 'training_right_labels.npy'))
training_index_right = np.load(os.path.join(training_dataset_path, 'training_right_indexes.npy'))
image_base_path_training_right = os.path.join(training_dataset_path, 'images\\right')


validation_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation"
validation_labels = np.load(os.path.join(validation_dataset_path, 'validation_center_labels.npy'))
validation_index_center = np.load(os.path.join(validation_dataset_path, 'validation_center_indexes.npy'))
image_base_path_validation = os.path.join(validation_dataset_path, 'images\\center')

zeros = np.zeros_like(validation_labels)
rmse_test = np.sqrt(np.mean(np.square(validation_labels - zeros)))
print(rmse_test)

zeros = np.zeros_like(training_labels_center)
rmse_test = np.sqrt(np.mean(np.square(training_labels_center - zeros)))
print(rmse_test)


'''
for val in keras_help.generate_arrays_from_file_v4((training_labels_center,training_labels_left, training_labels_right), (training_index_center, training_index_left,
                    training_index_right), (image_base_path_training_center, image_base_path_training_left, image_base_path_training_right), 1):
    image, y = val
    print(image)
    print(y)
    print(image.shape)
    plt.imshow(image[0])
'''

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

model.fit_generator(keras_help.generate_arrays_from_file_v2(training_labels, training_index_center, image_base_path_training, 32),
                    steps_per_epoch=training_labels.shape[0] // 32, epochs=10, verbose=1)

model.fit_generator(keras_help.generate_arrays_from_file_v2(training_labels, training_index_center, image_base_path_training, 32),
                    steps_per_epoch=training_labels.shape[0] // 32,
                    validation_data=keras_help.generate_arrays_from_file_v2(validation_labels, validation_index_center, image_base_path_validation, 32),
                    validation_steps=validation_labels.shape[0] // 32, epochs=40, verbose=1)
'''
'''
model = Sequential()
model.add(Conv2D(24, (5, 5), strides=2, activation='relu', input_shape=(240, 320, 3)))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.layers[-1].output_shape)
model.add(Conv2D(36, (5, 5), strides=2, activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.layers[-1].output_shape)
model.add(Conv2D(48, (3, 3), strides=2, activation='relu'))
model.add(BatchNormalization())
print(model.layers[-1].output_shape)
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.layers[-1].output_shape)
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
print(model.layers[-1].output_shape)
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[keras_help.rmse])
'''
'''
model.fit_generator(keras_help.generate_arrays_from_file_v2(training_labels, training_index_center, image_base_path_training, 256),
                    steps_per_epoch=training_labels.shape[0] // 256,
                    validation_data=keras_help.generate_arrays_from_file_v2_val(validation_labels, validation_index_center, image_base_path_validation, 32),
                    validation_steps=validation_labels.shape[0] // 32, epochs=100, verbose=1)
'''
'''
model = Sequential()
model.add(Conv2D(3, (1, 1), strides=2, activation='relu', input_shape=(240, 320, 3)))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.layers[-1].output_shape)
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(SpatialDropout2D(0.1))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(SpatialDropout2D(0.1))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(SpatialDropout2D(0.1))

print(model.layers[-1].output_shape)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[keras_help.rmse])
'''
model = Sequential()
model.add(Conv2D(16, 5, 5,
                 input_shape=(240, 320, 3),
                 init="he_normal",
                 activation='relu',
                 border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, 5, 5,
                 init="he_normal",
                 activation='relu',
                 border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(40, 3, 3,
                 init="he_normal",
                 activation='relu',
                 border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(60, 3, 3,
                 init="he_normal",
                 activation='relu',
                 border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(80, 2, 2,
                 init="he_normal",
                 activation='relu',
                 border_mode='same'))
model.add(SpatialDropout2D(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, 2, 2,
                 init="he_normal",
                 activation='relu',
                 border_mode='same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(
    output_dim=1,
    init='he_normal',
    W_regularizer=l2(0.0001)))

#optimizer = (SGD(lr=.001, momentum=0.9))
optimizer = 'adadelta'
model.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    metrics=[keras_help.rmse])

print(keras_help.std_evaluate(model, keras_help.generate_arrays_from_file_v2_val(validation_labels, validation_index_center, image_base_path_validation, 32), 32))
model.fit_generator(keras_help.generate_arrays_from_file_v2(training_labels_center, training_index_center, image_base_path_training_center, 32),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=keras_help.generate_arrays_from_file_v2_val(validation_labels, validation_index_center, image_base_path_validation, 32),
                    validation_steps=validation_labels.shape[0] // 32, epochs=32, verbose=1)
'''
model.fit_generator(keras_help.generate_arrays_from_file_v4((training_labels_center,training_labels_left, training_labels_right), (training_index_center, training_index_left,
                    training_index_right), (image_base_path_training_center, image_base_path_training_left, image_base_path_training_right), 32),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=keras_help.generate_arrays_from_file_v2_val(validation_labels, validation_index_center, image_base_path_validation, 32),
                    validation_steps=validation_labels.shape[0] // 32, epochs=32, verbose=1)
'''

print(keras_help.std_evaluate(model, keras_help.generate_arrays_from_file_v2_val(validation_labels, validation_index_center, image_base_path_validation, 32), 32))




