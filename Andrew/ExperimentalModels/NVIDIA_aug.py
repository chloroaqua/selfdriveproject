import numpy as np
import tensorflow as tf
import keras
import os
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense, BatchNormalization
from keras.losses import mean_squared_error
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

import keras_help


model = load_model('nvidia_aug_v4_light_v2.h5', custom_objects={'rmse': keras_help.rmse})
#model = model_from_json(open('nvidia_no_aug.h5').read())
#model.load_weights(os.path.join(os.path.dirname('nvidia_no_aug'), 'model_weights.h5'))


training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
training_labels_center = np.load(os.path.join(training_dataset_path, 'training_center_labels.npy'))
training_index_center = np.load(os.path.join(training_dataset_path, 'training_center_indexes.npy'))
image_base_path_training_center = os.path.join(training_dataset_path, 'images\\center')

validation_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation"
validation_labels = np.load(os.path.join(validation_dataset_path, 'validation_center_labels.npy'))
validation_index_center = np.load(os.path.join(validation_dataset_path, 'validation_center_indexes.npy'))
image_base_path_validation = os.path.join(validation_dataset_path, 'images\\center')



print(keras_help.std_evaluate(model, keras_help.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), 32))

