import os

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from project_code.submission import overlay_angle, generate_video

# generate_video('M:\\selfdrive\\SelfDrivingData\\test_vid_3d\\3dconvlstm_test.csv', 'M:\\selfdrive\\SelfDrivingData\\export_ch2_001\\center',
#                    'M:\\selfdrive\SelfDrivingData\\test_vid_3d\\testvid.mp4',
#                    'M:\\selfdrive\\SelfDrivingData\\test_vid_3d\\tmp')

generate_video('M:\\selfdrive\\SelfDrivingData\\test_vid\\resnet50tran_best_test.csv', 'M:\\selfdrive\\SelfDrivingData\\export_ch2_001\\center',
                   'M:\\selfdrive\SelfDrivingData\\test_vid\\testvid.mp4',
                   'M:\\selfdrive\\SelfDrivingData\\test_vid\\tmp')


# data = pd.read_csv('resnet50transv2.csv', dtype=object)
# frames = [1479425825597986783, 1479425066798657538, 1479425966120917767, 1479424714486951477, 1479426034382842936, 1479425885606859507, 1479425937165981415, 1479425941366566287, 1479425945417365320]
# for image_index, (time_stamp, row) in enumerate(data.iterrows()):
#     filename = row.real[0]
#     angle_predict = row.real[1]
#     angle_actual = row.real[2]
#     if int(filename) in frames:
#         print(filename, angle_predict, angle_actual)

data = pd.read_csv('resnet50tran_best_test.csv', dtype=object)
frames = [1479425498692668616, 1479425533848771224, 1479425542300264703, 1479425552051985269, 1479425598310084542, 1479425659820741440]
# data = pd.read_csv('resnet50transv2.csv', dtype=object)
# frames = [1479425825597986783, 1479425066798657538, 1479425966120917767, 1479424714486951477, 1479426034382842936, 1479425885606859507, 1479425937165981415, 1479425941366566287, 1479425945417365320]
for image_index, (time_stamp, row) in enumerate(data.iterrows()):
    filename_time = row.real[0]
    angle_predict = row.real[1]
    angle_actual = row.real[2]
    if int(filename_time) in frames:
        filename = 'M:\\selfdrive\\SelfDrivingData\\export_ch2_001\\center\\{}.jpg'.format(filename_time)
        image = overlay_angle(filename, float(angle_predict), float(angle_actual))
        plt.imshow(image)

    #print(row.real[0])

#
# image = np.load("M:\\selfdrive\\SelfDrivingData\\test_out2\\training\\images\\center\\10001.jpg.npy")
# image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
# image = image.astype('uint8')
# image_copy = np.copy(image)
#
# image = overlay_angle('M:\\selfdrive\\SelfDrivingData\\export_ch2_002\\center\\1479424276243601127.jpg', 0.03)
# image = image.astype('uint8')
# plt.imshow(image)
# print("test")

#1479425498692668616
#1479425533848771224
#1479425542300264703
#1479425552051985269
#1479425598310084542
#1479425659820741440


'''
1479425825597986783.jpg, 
1479425066798657538.jpg, 
1479425966120917767.jpg, 
1479424714486951477.jpg, 
1479426034382842936.jpg, 
1479425885606859507.jpg, 
1479425937165981415.jpg, 
1479425941366566287.jpg, 
1479425945417365320.jpg
'''