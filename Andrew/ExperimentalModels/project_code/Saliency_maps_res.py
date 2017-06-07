import os

import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt, cm

from vis.visualization import visualize_saliency

from old import keras_help
from project_code import util

#model = load_model('../nvidia_no_aug_v2.h5', custom_objects={'rmse': keras_help.rmse})
model = load_model('../models/res50_trans_net_test.h5', custom_objects={'rmse': keras_help.rmse})
#model = model_from_json(open('nvidia_no_aug.h5').read())
#model.load_weights(os.path.join(os.path.dirname('nvidia_no_aug'), 'model_weights.h5'))



for idx, layer in enumerate(model.layers):
    print(idx, layer)

for layer in model.layers:
   layer.trainable = True


data, image_copies = util.get_images_single_res("M:\\selfdrive\\SelfDrivingData\\test_out3\\training\\images\\center\\", 1603)
#data, image_copies = util.get_images_single("M:\\selfdrive\\SelfDrivingData\\test_out2\\training\\images\\center\\", 10000)
#nvidia 11
#res 6
pred_value = model.predict(data)
heatmap, grads = visualize_saliency(model, 6, [0], data)

print(grads.shape)
grads_new = np.sum(grads, axis=0)
heatmap_new = np.uint8(cm.jet(grads_new)[..., :3] * 255)
heatmap_new_no_aug = np.uint8(image_copies[0] * .5 + heatmap_new * (1. - .5))


plt.imshow(heatmap_new_no_aug)
plt.title('Transfer Model')
plt.axis('off')


plt.savefig("transfer_test.png")
plt.show()





# print(keras_help.std_evaluate(model, keras_help.generate_arrays_from_file_new(validation_labels, validation_index_center, image_base_path_validation, 32), 32))
print('Model loaded.')
#
# # The name of the layer we want to visualize
# # (see model definition in vggnet.py)
# layer_name = 'predictions'
# layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
#
#





