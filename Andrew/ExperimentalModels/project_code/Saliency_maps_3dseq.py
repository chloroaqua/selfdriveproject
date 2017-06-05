import os

import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt, animation, cm

from project_code import vis, util
from project_code.vis.visualization import visualize_saliency
from keras import backend as K
import tensorflow as tf

model = load_model('../models/project_model_3dconv_lstm_best.h5', custom_objects={'rmse': util.rmse})
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
batch_size = 1

#get_images_seq(start_image_path, number_of_frames, seq_length):
data, image_copies = util.get_images_seq('M:\\selfdrive\\SelfDrivingData\\test_out2\\training\\images\\center\\', seq_frames, num_seqs, 5628)

y_pred = model.predict(data)
print(y_pred)

layer_dict = dict([(layer.name, layer) for layer in model.layers])
# #layer_name = 'block5_conv3'
# #layer_output = layer_dict[layer_name].output
# layer_output = model.layers[28].output
# loss = K.mean(layer_output)
# grads = K.gradients(layer_output, data)[0]
#
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
# iterate = K.function([data], [loss, grads])
#
# # outputTensor = model.output
# # variableTensors = model.trainable_weights[0]
# # grads_fn = K.gradients(loss_fn, input_tensor)[0]
# # loss_grads_fn =K.function([input_tensor], [loss_fn, grads_fn])


heatmap, grads = visualize_saliency(model, 28, [0], data)
print(grads.shape)
grads_new = np.sum(np.abs(grads), axis=(0,1,2))
grads_new /= np.max(grads_new)
# heatmap_new = np.uint8(cm.jet(grads_new)[..., :3] * 255)
# heatmap_new = np.uint8(image_copies[0] * .5 + heatmap_new * (1. - .5))
heatmap_new = np.uint8(cm.jet(grads_new)[..., :3] * 255)
heatmap_new = np.uint8(image_copies[0][0][0] * .5 + heatmap_new * (1. - .5))


print(training_labels_center[10000])
print(heatmap.shape)
plt.imshow(heatmap_new)
plt.axis('off')
plt.title("3D Convolutional Model with LSTM (Collapsed)")
plt.show()

# plt.figure(figsize=(12, 6))
# for i in range(5):
#     plt.subplot(5, 1, i + 1)
#     plt.imshow(heatmap_new[0][i])
#     plt.axis('off')
#
# plt.gcf().tight_layout()
# plt.show()


print("test")





