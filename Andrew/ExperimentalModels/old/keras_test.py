import os

import numpy as np

from old import keras_help

dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
labels = np.load(os.path.join(dataset_path, 'training_center_indexes_labels.npy'))
training_index_center = np.load(os.path.join(dataset_path, 'training_center_indexes.npy'))
image_base_path = os.path.join(dataset_path, 'images\\center')
for val in keras_help.generate_arrays_from_file(labels, training_index_center, image_base_path):
    image, y = val
    print(image)
    print(y)





def load_data(image_index, image_base_path, size):
    batch_value = np.zeros(size)
    for i in image_index:
        image_path = os.path.join(image_base_path, "{}.jpg.npy".format(i))
        image = np.load(image_path)
        batch_value[i] = image
    return batch_value




dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"


labels = np.load(os.path.join(dataset_path, 'training_center_indexes_labels.npy'))
print(labels)
training_index_center = np.load(os.path.join(dataset_path, 'training_center_indexes.npy'))
print(training_index_center)

image_base_path = os.path.join(dataset_path, 'images\\center')

image_path = os.path.join(image_base_path, "{}.jpg.npy".format("0"))
image = np.load(image_path)
#plt.imshow(image)
print(image.shape)
sample_value = training_index_center[:1000]
batch_value = load_data(sample_value, image_base_path, (sample_value.size, *image.shape))







