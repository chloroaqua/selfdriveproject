import os

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from project_code.submission import overlay_angle

image = np.load("M:\\selfdrive\\SelfDrivingData\\test_out2\\training\\images\\center\\10001.jpg.npy")
image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
image = image.astype('uint8')
image_copy = np.copy(image)

image = overlay_angle('M:\\selfdrive\\SelfDrivingData\\export_ch2_002\\center\\1479424276243601127.jpg', 0.03)
image = image.astype('uint8')
plt.imshow(image)
print("test")

