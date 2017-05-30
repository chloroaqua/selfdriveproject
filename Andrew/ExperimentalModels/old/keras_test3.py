import os

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from keras.models import Sequential
from keras.regularizers import l2

from old import keras_help

training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
training_labels_center = np.load(os.path.join(training_dataset_path, 'training_center_labels.npy'))
training_index_center = np.load(os.path.join(training_dataset_path, 'training_center_indexes.npy'))
image_base_path_training_center = os.path.join(training_dataset_path, 'images\\center')

validation_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation"
validation_labels = np.load(os.path.join(validation_dataset_path, 'validation_center_labels.npy'))
validation_index_center = np.load(os.path.join(validation_dataset_path, 'validation_center_indexes.npy'))
image_base_path_validation = os.path.join(validation_dataset_path, 'images\\center')


rmse_test = np.sqrt(np.mean(np.square(validation_labels)))
print(rmse_test)

rmse_test = np.sqrt(np.mean(np.square(training_labels_center)))
print(rmse_test)


# for val in keras_help.generate_arrays_from_file_new(training_labels_center, training_index_center, image_base_path_training_center, 32):
#     image, y = val
#     print(image)
#     print(y)
#     print(image.shape)
#     plt.imshow(image[0])


model = Sequential()
model.add(Conv2D(16, 5, 5,
                 input_shape=(120, 320, 3),
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
#generate_arrays_from_file_new(labels, index_values, image_path_base, batch_size)
print(keras_help.std_evaluate(model, keras_help.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), 32))

'''
model.fit_generator(keras_help.generate_arrays_from_file_new(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=keras_help.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1),
                    validation_steps=validation_labels.shape[0] // 32, epochs=40, verbose=1)
'''
model.fit_generator(keras_help.generate_arrays_from_file_new_exp(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=keras_help.generate_arrays_from_file_new_exp(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1),
                    validation_steps=validation_labels.shape[0] // 32, epochs=40, verbose=1)
keras_help.generate_arrays_from_file_new_exp(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1)

print(keras_help.std_evaluate(model, keras_help.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), 32))




