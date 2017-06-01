import os

import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt, animation

from project_code import vis
from project_code.vis.visualization import visualize_saliency
import project_code.util as ut

model = load_model('../models/3d_test_v3.h5', custom_objects={'rmse': ut.rmse})
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

'''
image = np.load('M:\\selfdrive\\SelfDrivingData\\test_out2\\training\\images\\center\\16046.jpg.npy')
cv_image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
plt.imshow(cv_image.astype('uint8'))
'''

data, image_copies = ut.get_images("M:\\selfdrive\\SelfDrivingData\\test_out2\\training\\images\\center\\", 10, 10000)


pred_value = model.predict(data)
heatmap = visualize_saliency(model, 1, [10], image_copies)
print(training_labels_center[10000])
#plt.imshow(vis.utils.stitch_images(heatmap))
# plt.imshow(heatmap[0])
# plt.imshow(heatmap[1])
# plt.imshow(heatmap[2])
# plt.title('Saliency map')
# plt.show()

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(heatmap[i])
    plt.axis('off')
plt.gcf().tight_layout()
plt.show()

print(ut.std_evaluate(model, ut.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), 32))
print('Model loaded.')

# The name of the layer weplt.imshow(utils.stitch_images(heatmaps)) want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]



model.predict(data)



