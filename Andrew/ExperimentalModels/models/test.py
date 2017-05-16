import os

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

print("test")
def make_np_data(input_path, output_path, cam_location, type):
    cam_image_path = os.path.join(input_path, cam_location)
    images_output_path = os.path.join(output_path, 'images\\{}'.format(cam_location))
    try:
        os.makedirs(images_output_path)
    except:
        pass
    print(input_path)
    for file in os.listdir(input_path):
        absolute_file = os.path.join(input_path, str(file))
        print(absolute_file)
    sensor_csv_path = os.path.join(input_path, 'interpolated.csv')
    sensor_df = pd.DataFrame.from_csv(sensor_csv_path)
    part_dfs = []
    center_df = sensor_df[sensor_df['frame_id'] == '{}_camera'.format(cam_location)].copy()
    part_dfs.append(center_df[['timestamp', 'filename', 'angle']])
    master_df = pd.concat(part_dfs).sort_values('timestamp')
    n_original_samples = len(master_df)
    labels = np.empty(n_original_samples)

    indexes = np.arange(0, n_original_samples)
    np.save(os.path.join(output_path, '{}_{}_indexes.npy'.format(type, cam_location)), indexes)
    np.save(os.path.join(output_path, '{}_{}_labels.npy'.format(type, cam_location)), labels)
    for image_index, (_, row) in enumerate(master_df.iterrows()):
        if image_index % 1000 == 0:
            print(image_index)
        # print(image_index, row)
        current_out_filename = os.path.join(images_output_path, '%d.jpg.npy' % image_index)
        labels[image_index] = row.angle
        current_image = os.path.join(input_path, str(row.filename))
        cv_image = cv2.imread(current_image)
        cv_image = cv2.resize(cv_image, (320, 240))
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
        #print(current_out_filename)
        np.save(current_out_filename, cv_image)
        #plt.imshow(cv_image)

    #np.save(os.path.join(output_path, 'labels_{}_{}.npy'.format(type, cam_location)), labels)

make_np_data("M:\\selfdrive\\SelfDrivingData\\export_ch2_002", "M:\\selfdrive\\SelfDrivingData\\test_out2\\training", "center", "training")
#make_np_data("M:\\selfdrive\\SelfDrivingData\\export_ch2_002", "M:\\selfdrive\\SelfDrivingData\\test_out2\\training", "left", "training")
#make_np_data("M:\\selfdrive\\SelfDrivingData\\export_ch2_002", "M:\\selfdrive\\SelfDrivingData\\test_out2\\training", "right", "training")

#make_np_data("M:\\selfdrive\\SelfDrivingData\\export_hmb_3", "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation", "center", "validation")
#make_np_data("M:\\selfdrive\\SelfDrivingData\\export_hmb_3", "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation", "left", "validation")
#make_np_data("M:\\selfdrive\\SelfDrivingData\\export_hmb_3", "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation", "right", "validation")



