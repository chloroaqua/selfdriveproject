import os

import numpy as np
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential

from old import keras_help

training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
training_labels_center = np.load(os.path.join(training_dataset_path, 'training_center_labels.npy'))
training_index_center = np.load(os.path.join(training_dataset_path, 'training_center_indexes.npy'))
image_base_path_training_center = os.path.join(training_dataset_path, 'images\\center')

validation_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation"
validation_labels = np.load(os.path.join(validation_dataset_path, 'validation_center_labels.npy'))
validation_index_center = np.load(os.path.join(validation_dataset_path, 'validation_center_indexes.npy'))
image_base_path_validation = os.path.join(validation_dataset_path, 'images\\center')

'''
for val in keras_help.generate_arrays_from_file_new_augment(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1):
    image, y = val
    print(image)
    print(y)
    print(image.shape)
    plt.imshow((image[0]*255+180).astype('uint8'))
    #plt.imshow(image[0]*255+180)
'''

rmse_test = np.sqrt(np.mean(np.square(validation_labels)))
print(rmse_test)

rmse_test = np.sqrt(np.mean(np.square(training_labels_center)))
print(rmse_test)

model = Sequential()
model.add(Conv2D(3, (5, 5), strides=2, activation='relu', input_shape=(120, 320, 3)))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.layers[-1].output_shape)
model.add(Conv2D(24, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(36, (5, 5), strides=2, activation='relu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.layers[-1].output_shape)
model.add(Conv2D(48, (3, 3), strides=2, activation='relu'))
#model.add(BatchNormalization())
print(model.layers[-1].output_shape)
model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.layers[-1].output_shape)
model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(BatchNormalization())
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
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras_help.rmse])

print(keras_help.std_evaluate(model, keras_help.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), 32))


model.fit_generator(
    keras_help.generate_arrays_from_file_new_augment_light(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1),
    steps_per_epoch=training_labels_center.shape[0] // 32,
    validation_data=keras_help.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32, scale=1),
    validation_steps=validation_labels.shape[0] // 32, epochs=40, verbose=1)

'''

model.fit_generator(keras_help.generate_arrays_from_file_new_exp(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1),
                    steps_per_epoch=training_labels_center.shape[0] // 32,
                    validation_data=keras_help.generate_arrays_from_file_new_exp(training_labels_center, training_index_center, image_base_path_training_center, 32, scale=1, val=True),
                    validation_steps=validation_labels.shape[0] // 32, epochs=40, verbose=1)
'''

model.save('nvidia_aug_v4_light_v2.h5')

print(keras_help.std_evaluate(model, keras_help.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), 32))

