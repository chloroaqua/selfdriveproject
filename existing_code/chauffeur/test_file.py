#from callbacks import SnapshotCallback
#from datasets import load_dataset, prepare_final_dataset
#from models import load_from_config, RegressionModel
import numpy as np
import logging
import multiprocessing
import os

import cv2
import pandas as pd

logger = logging.getLogger(__name__)

'''
def prepare_final_dataset(
        local_raw_path,
        local_output_path,
        training_percent=0.7,
        testing_percent=0.2,
        validation_percent=0.1):
'''

def prepare_final_dataset(
        local_raw_path,
        local_output_path,
        training_percent=0.7,
        testing_percent=0.2,
        validation_percent=0.1):
    train_path = os.path.join(local_raw_path, 'Train')

    # ensure images path exists
    images_path = os.path.join(local_output_path, 'images')
    logger.info('Using %s as base images directory', images_path)
    try: os.makedirs(images_path)
    except: pass

    part_dfs = []
    for part_no in os.listdir(train_path):
        part_path = os.path.join(train_path, str(part_no))
        sensor_csv_path = os.path.join(part_path, 'interpolated.csv')
        sensor_df = pd.DataFrame.from_csv(sensor_csv_path)
        center_df = sensor_df[sensor_df['frame_id'] == 'center_camera'].copy()
        center_df['filename'] = (
            (part_path + '/') + center_df.filename.astype(str))

        part_dfs.append(center_df[['timestamp', 'filename', 'angle']])

    # concat all the path directory csvs
    master_df = pd.concat(part_dfs).sort_values('timestamp')

    n_original_samples = len(master_df)
    n_samples = len(master_df) * 2
    n_training = int(training_percent * n_samples)
    n_testing = int(testing_percent * n_samples)
    n_validation = n_samples - n_training - n_testing

    logger.info('%d total samples in the dataset', n_samples)
    logger.info('%d samples in training set', n_training)
    logger.info('%d samples in testing set', n_testing)
    logger.info('%d samples in validation set', n_validation)

    labels = np.empty(n_samples)
    tasks = []
    for image_index, (_, row) in enumerate(master_df.iterrows()):
        labels[image_index] = row.angle
        labels[image_index + n_original_samples] = -row.angle
        tasks.append(
            (row.filename, images_path, image_index + 1, image_index + n_original_samples + 1))

    indexes = np.arange(1, n_samples + 1)
    np.random.shuffle(indexes)

    training_indexes = indexes[:n_training]
    testing_indexes = indexes[n_training:(n_training + n_testing)]
    validation_indexes = indexes[-n_validation:]

    np.save(os.path.join(local_output_path, 'labels.npy'), labels)
    np.save(
        os.path.join(local_output_path, 'training_indexes.npy'),
        training_indexes)
    np.save(
        os.path.join(local_output_path, 'testing_indexes.npy'),
        testing_indexes)
    np.save(
        os.path.join(local_output_path, 'validation_indexes.npy'),
        validation_indexes)
    if __name__ == '__main__':
        pool = multiprocessing.Pool(4)
        pool.map(process_final_image, tasks)


def process_final_image(args):
    src_path, dest_dir, image_index, flipped_image_index = args
    normal_path = os.path.join(dest_dir, '%d.png.npy' % image_index)
    flipped_path = os.path.join(dest_dir, '%d.png.npy' % flipped_image_index)

    cv_image = cv2.imread(src_path)
    cv_image = cv2.resize(cv_image, (320, 240))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
    cv_image = cv_image[120:240, :, :]

    np.save(normal_path, cv_image)

    # flip the image over the y axis to equalize left/right turns
    cv_image = cv_image[:, ::-1, :]
    np.save(flipped_path, cv_image)


prepare_final_dataset("M:\\selfdrive\\SelfDrivingData\\test", "M:\\selfdrive\SelfDrivingData\\test_out")

'''
# RegressionModel is a cnn that directly predicts the steering angle
init_model_config = RegressionModel.create(
    '/tmp/regression_model.keras',
    use_adadelta=True,
    learning_rate=0.001,
    input_shape=(120, 320, 3))

model = load_from_config(init_model_config)

# this path contains a dataset in the prescribed format
dataset = load_dataset('/datasets/showdown_full')

# snapshots the model after each epoch
snapshot = SnapshotCallback(
    model,
    snapshot_dir='/tmp/snapshots/',
    score_metric='val_rmse')

model.fit(dataset, {
    'batch_size': 32,
    'epochs': 40,
},
          final=False,  # don't train on the test holdout set
          callbacks=[snapshot])

# save model to local file and return the 'config' so it can be loaded
model_config = model.save('/tmp/regression.keras')

# evaluate the model on the test holdout
print()
model.evaluate(dataset)

### Generating a video on the test set with overlayed steering anle
'''