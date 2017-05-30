import keras
import numpy as np
import os
import keras.backend as K
import scipy.ndimage
import random
import matplotlib.pyplot as plt

image_path = 'M:\\selfdrive\\SelfDrivingData\\test_out2\\training\\images\\center\\33299.jpg.npy'
image = np.load(image_path)
print(image.shape)
plt.imshow(image)
print(image.shape)
