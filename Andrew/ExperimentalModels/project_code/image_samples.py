import os

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

from project_code.submission import overlay_angle
from project_code.util import augment_brightness_camera_images, add_random_shadow, trans_image

image = np.load("M:\\selfdrive\\SelfDrivingData\\test_out2\\training\\images\\center\\1359.jpg.npy")
image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
image = image.astype('uint8')
image_copy = np.copy(image)

#image, test = trans_image(image, 2, 50)
image = add_random_shadow(image)
image = augment_brightness_camera_images(image)

image = scipy.ndimage.interpolation.rotate(image, scipy.random.uniform(-15, 15), reshape=False)
#plt.imshow(image)
print("test")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_copy)
plt.axis('off')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(image)
plt.axis('off')

plt.title('Augmented Example')
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(heatmap[i])
#     plt.axis('off')
#plt.gcf().tight_layout()
plt.show()
plt.savefig("augmented_example.png", bbox_inches='tight')

print("test")

