import os

import pandas as pd
import numpy as np
from keras.models import load_model
from progress.bar import IncrementalBar
import cv2

import csv

from project_code import util
from keras.utils import plot_model


def load_images_from_file_seq_test(input_path, output_file, model, cam_location, numpy_path):

    test_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\test"

    sensor_csv_path = os.path.join(input_path, 'CH2_final_evaluation.csv')
    sensor_df = pd.read_csv(sensor_csv_path, dtype=object)
    master_df = sensor_df.sort_values('frame_id')
    n_original_samples = len(master_df)

    indexes = np.arange(0, n_original_samples)
    # np.save(os.path.join(output_path, '{}_{}_indexes.npy'.format(type, cam_location)), indexes)
    # np.save(os.path.join(output_path, '{}_{}_labels.npy'.format(type, cam_location)), labels)
    data = []
    for image_index, (time_stamp, row) in enumerate(master_df.iterrows()):
        if image_index % 100 == 0:
            print(image_index)
        #imageidx = training_labels_center[image_index]
        image_data = np.zeros((1, 5, 5, 120, 320, 3))
        for seq in range(5):
            for frame in range(5):
                current_value = image_index + frame + seq
                if current_value >= len(indexes):
                    current_value = image_index
                image_path = os.path.join(numpy_path, "{}.jpg.npy".format(int(current_value)))
                image = np.load(image_path)
                image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                image = ((image - (255.0 / 2)) / 255.0)
                image_data[0, seq, frame, :] = image
        # if image_index == 25:
        #     print("25")
        angle = row.steering_angle
        angle_predict = model.predict(image_data)[0][0][0]
        data.append((row.frame_id, angle_predict, angle))
        #print(time_stamp, angle_predict)



        # print(current_out_filename)

        #np.save(current_out_filename, cv_image)
        # plt.imshow(cv_image)

    with open(output_file, 'w') as file:
        for line in data:
            file.write(str(line[0]))
            file.write(",")
            file.write(str(line[1]))
            file.write(",")
            file.write(str(line[2]))
            file.write('\n')


def load_images_from_file_test_set(input_path, output_file, model, res_size=True):
    sensor_csv_path = os.path.join(input_path, 'CH2_final_evaluation.csv')
    sensor_df = pd.read_csv(sensor_csv_path, dtype=object)
    part_dfs = []
    # #center_df = sensor_df[sensor_df['frame_id'] == '{}_camera'.format(cam_location)].copy()
    #part_dfs.append(sensor_df[['frame_id', 'steering_angle', 'public']])
    master_df = sensor_df.sort_values('frame_id')
    n_original_samples = len(master_df)
    #labels = np.empty(n_original_samples)

    indexes = np.arange(0, n_original_samples)
    # np.save(os.path.join(output_path, '{}_{}_indexes.npy'.format(type, cam_location)), indexes)
    # np.save(os.path.join(output_path, '{}_{}_labels.npy'.format(type, cam_location)), labels)
    data = []
    for image_index, (time_stamp, row) in enumerate(master_df.iterrows()):
        if image_index % 100 == 0:
            print(image_index)
        # if image_index == 3:
        #     break
        # print(image_index, row)
        #current_out_filename = os.path.join(images_output_path, '%d.jpg.npy' % image_index)
        angle = row.steering_angle
        #labels[image_index] = angle
        current_image = os.path.join(input_path, 'center\\'+str(row.frame_id)+".jpg")
        cv_image = cv2.imread(current_image)
        if res_size:
            # cv_image = cv2.resize(cv_image, (320, 240))
            cv_image = cv2.resize(cv_image, (224, 224))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
            # cv_image = cv_image[120:240, :, :]
            cv_image[:, :, 0] = cv2.equalizeHist(cv_image[:, :, 0])
            image = ((cv_image - (255.0 / 2)) / 255.0)
            # angle_predict = model.predict(np.reshape(image, (1, 120, 320, 3)))[0][0]
            angle = row.steering_angle
            angle_predict = model.predict(np.reshape(image, (1, 224, 224, 3)))[0][0]
        else:
            cv_image = cv2.resize(cv_image, (320, 240))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
            cv_image = cv_image[120:240, :, :]
            cv_image[:, :, 0] = cv2.equalizeHist(cv_image[:, :, 0])
            image = ((cv_image - (255.0 / 2)) / 255.0)
            angle_predict = model.predict(np.reshape(image, (1, 120, 320, 3)))[0][0]
            angle = row.steering_angle

        data.append((row.frame_id, angle_predict, angle))
        #print(time_stamp.value, angle_predict, image_index)

        # print(current_out_filename)

        #np.save(current_out_filename, cv_image)
        # plt.imshow(cv_image)

    with open(output_file, 'w') as file:
        for line in data:
            file.write(str(line[0]))
            file.write(",")
            file.write(str(line[1]))
            file.write(",")
            file.write(str(line[2]))
            file.write('\n')

def load_images_from_file(input_path, output_file, model, cam_location):
    sensor_csv_path = os.path.join(input_path, 'interpolated.csv')
    sensor_df = pd.DataFrame.from_csv(sensor_csv_path)
    part_dfs = []
    center_df = sensor_df[sensor_df['frame_id'] == '{}_camera'.format(cam_location)].copy()
    part_dfs.append(center_df[['timestamp', 'filename', 'angle']])
    master_df = pd.concat(part_dfs).sort_values('timestamp')
    n_original_samples = len(master_df)
    labels = np.empty(n_original_samples)

    indexes = np.arange(0, n_original_samples)
    # np.save(os.path.join(output_path, '{}_{}_indexes.npy'.format(type, cam_location)), indexes)
    # np.save(os.path.join(output_path, '{}_{}_labels.npy'.format(type, cam_location)), labels)
    data = []
    for image_index, (time_stamp, row) in enumerate(master_df.iterrows()):
        if image_index % 100 == 0:
            print(image_index)
        # if image_index == 3:
        #     break
        # print(image_index, row)
        #current_out_filename = os.path.join(images_output_path, '%d.jpg.npy' % image_index)
        angle = row.angle
        labels[image_index] = angle
        current_image = os.path.join(input_path, str(row.filename))
        cv_image = cv2.imread(current_image)
        #cv_image = cv2.resize(cv_image, (320, 240))
        cv_image = cv2.resize(cv_image, (224, 224))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
        #cv_image = cv_image[120:240, :, :]
        cv_image[:, :, 0] = cv2.equalizeHist(cv_image[:, :, 0])
        image = ((cv_image - (255.0 / 2)) / 255.0)
        #angle_predict = model.predict(np.reshape(image, (1, 120, 320, 3)))[0][0]
        angle = row.angle
        angle_predict = model.predict(np.reshape(image, (1, 224, 224, 3)))[0][0]
        data.append((time_stamp.value, angle_predict, angle))
        #print(time_stamp.value, angle_predict, image_index)

        # print(current_out_filename)

        #np.save(current_out_filename, cv_image)
        # plt.imshow(cv_image)

    with open(output_file, 'w') as file:
        for line in data:
            file.write(str(line[0]))
            file.write(",")
            file.write(str(line[1]))
            file.write(",")
            file.write(str(line[2]))
            file.write('\n')


def load_images_from_file_seq(input_path, output_file, model, cam_location, numpy_path):
    training_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\training"
    training_labels_center = np.load(os.path.join(training_dataset_path, 'training_center_labels.npy'))
    training_index_center = np.load(os.path.join(training_dataset_path, 'training_center_indexes.npy'))
    image_base_path_training_center = os.path.join(training_dataset_path, 'images\\center')

    validation_dataset_path = "M:\\selfdrive\\SelfDrivingData\\test_out2\\validation"
    validation_labels = np.load(os.path.join(validation_dataset_path, 'validation_center_labels.npy'))
    validation_index_center = np.load(os.path.join(validation_dataset_path, 'validation_center_indexes.npy'))
    image_base_path_validation = os.path.join(validation_dataset_path, 'images\\center')

    sensor_csv_path = os.path.join(input_path, 'interpolated.csv')
    sensor_df = pd.DataFrame.from_csv(sensor_csv_path)
    part_dfs = []
    center_df = sensor_df[sensor_df['frame_id'] == '{}_camera'.format(cam_location)].copy()
    part_dfs.append(center_df[['timestamp', 'filename', 'angle']])
    master_df = pd.concat(part_dfs).sort_values('timestamp')
    n_original_samples = len(master_df)
    labels = np.empty(n_original_samples)

    indexes = np.arange(0, n_original_samples)
    # np.save(os.path.join(output_path, '{}_{}_indexes.npy'.format(type, cam_location)), indexes)
    # np.save(os.path.join(output_path, '{}_{}_labels.npy'.format(type, cam_location)), labels)
    data = []
    for image_index, (time_stamp, row) in enumerate(master_df.iterrows()):
        if image_index % 100 == 0:
            print(image_index)
        #imageidx = training_labels_center[image_index]
        image_data = np.zeros((1, 5, 5, 120, 320, 3))
        for seq in range(5):
            for frame in range(5):
                current_value = image_index + frame + seq
                if current_value >= len(training_index_center) :
                    current_value = image_index
                image_path = os.path.join(numpy_path, "{}.jpg.npy".format(int(current_value)))
                image = np.load(image_path)
                image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                image = ((image - (255.0 / 2)) / 255.0)
                image_data[0, seq, frame, :] = image
        # if image_index == 25:
        #     print("25")
        angle = row.angle
        angle_predict = model.predict(image_data)[0][0][0]
        data.append((time_stamp.value, angle_predict, angle))
        #print(time_stamp, angle_predict)



        # print(current_out_filename)

        #np.save(current_out_filename, cv_image)
        # plt.imshow(cv_image)

    with open(output_file, 'w') as file:
        for line in data:
            file.write(str(line[0]))
            file.write(",")
            file.write(str(line[1]))
            file.write(",")
            file.write(str(line[2]))
            file.write('\n')




#model = load_model('../models/res50_trans_net_test3_best.h5', custom_objects={'rmse': util.rmse})

#load_images_from_file_test_set('M:\\selfdrive\\SelfDrivingData\\export_ch2_001', 'resnet50tran_best_test.csv', model)


#model = load_model('../models/res50_trans_net_test.h5', custom_objects={'rmse': util.rmse})
#load_images_from_file("M:\\selfdrive\\SelfDrivingData\\export_ch2_002", 'resnet50transv2.csv', model, 'center')
#model = load_model('../models/res50_trans_net_test.h5', custom_objects={'rmse': util.rmse})
#load_images_from_file("M:\\selfdrive\\SelfDrivingData\\export_hmb_3", 'resnet50trans_val.csv', model, 'center')
#del model

#model = load_model('../models/project_model_3dconv_lstm_best.h5', custom_objects={'rmse': util.rmse})
#load_images_from_file_seq("M:\\selfdrive\\SelfDrivingData\\export_ch2_002", '3dconvlstmv3.csv', model, 'center', 'M:\\selfdrive\SelfDrivingData\\test_out2\\training\\images\\center')

#model = load_model('../models/project_model_3dconv_lstm_best.h5', custom_objects={'rmse': util.rmse})
#load_images_from_file_seq("M:\\selfdrive\\SelfDrivingData\\export_hmb_3", '3dconvlstm_val2.csv', model, 'center', 'M:\\selfdrive\SelfDrivingData\\test_out2\\validation\\images\\center')


# model = load_model('../models/project_model_3dconv_lstm_best.h5', custom_objects={'rmse': util.rmse})
# load_images_from_file_seq_test("M:\\selfdrive\\SelfDrivingData\\export_ch2_001", '3dconvlstm_test.csv', model, 'center', 'M:\\selfdrive\SelfDrivingData\\test_out2\\test\\images\\center')

# model = load_model('../nvidia_no_aug_v2.h5', custom_objects={'rmse': util.rmse})
# load_images_from_file_test_set('M:\\selfdrive\\SelfDrivingData\\export_ch2_001', 'nvidia_no_aug_test.csv', model, res_size=False)
model = load_model('../models/project_model_3dconv_lstm_best.h5', custom_objects={'rmse': util.rmse})
plot_model(model, to_file='3dlstmv2.png', show_shapes=True, show_layer_names=False)
