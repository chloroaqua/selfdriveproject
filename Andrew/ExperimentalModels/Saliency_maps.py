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
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency

import cv2
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


for idx, layer in enumerate(model.layers):
    print(idx, layer)


image = np.load("M:\\selfdrive\\SelfDrivingData\\test_out2\\training\\images\\center\\10000.jpg.npy")
image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
#plt.imshow(image)

image_pro = image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
x = ((image-(255.0/2))/255.0)
pred_value = model.predict(np.reshape(x, (1,120,320,3)))
heatmap = visualize_saliency(model, 1, [11], image)
plt.imshow(heatmap)

print(keras_help.std_evaluate(model, keras_help.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), 32))
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

path = ''
seed_img = utils.load_img(path, target_size=(120, 320, 3))



model.predict(x)



