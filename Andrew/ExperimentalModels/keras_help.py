import keras
import numpy as np
import os
import keras.backend as K
import scipy.ndimage
import random
import cv2

def generate_arrays_from_file(labels, index_values, image_path_base, batch_size):
    while True:
        #f = open(image_path)
        for idx, image_id in enumerate(index_values):
            y = labels[idx]
            image_path = os.path.join(image_path_base, "{}.jpg.npy".format(index_values[image_id]))
            image = np.load(image_path)
            yield image, y
        #f.close()


def generate_arrays_from_file_v2(labels, index_values, image_path_base, batch_size):
    batch_features = np.zeros((batch_size, 240, 320, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            idx = np.random.choice(len(labels), 1)
            y = labels[idx]
            image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx])))
            image = np.load(image_path)
            flip_bit = random.randint(0, 1)
            if flip_bit == 1:
                image = np.flip(image, 1)
                y=y*-1
            #noise = np.random.normal(0, 25, image.shape[0]*image.shape[1]*image.shape[2]).reshape(image.shape)
            image = scipy.ndimage.interpolation.rotate(image, random.uniform(0, 360), reshape=False)
            rows, cols, chan = image.shape
            M = np.float32([[1, 0, random.randint(-50, 50)], [0, 1, random.randint(-50, 50)]])
            R = random.uniform(.95, 1.05)
            G = random.uniform(.95, 1.05)
            B = random.uniform(.95, 1.05)
            image = np.float_(image)
            image[:, :, 0] *= R
            image[:, :, 1] *= G
            image[:, :, 2] *= B
            image = cv2.warpAffine(image, M, (cols, rows))
            batch_features[i,:] = image
            batch_labels[i] = y
        yield batch_features, batch_labels


def generate_arrays_from_file_v2_val(labels, index_values, image_path_base, batch_size):
    batch_features = np.zeros((batch_size, 240, 320, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            idx = np.random.choice(len(labels), 1)
            y = labels[idx]
            image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx])))
            image = np.load(image_path)
            batch_features[i] = image
            batch_labels[i] = y
        yield batch_features, batch_labels


def load_image(image_index, image_base_path, size):
    batch_value = np.zeros(size)
    for i in image_index:
        image_path = os.path.join(image_base_path, "{}.jpg.npy".format(i))
        image = np.load(image_path)
        batch_value[i] = image
    return batch_value


#model.fit_generator(generate_arrays_from_file('./my_file.txt'),samples_per_epoch=10000,nb_epoch=10)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def std_evaluate(model, generator):
    """
    """
    size = generator.get_size()
    batch_size = generator.get_batch_size()
    n_batches = size // batch_size

    err_sum = 0.
    err_count = 0.
    for _ in range(n_batches):
        X_batch, y_batch = generator.next()
        y_pred = model.predict_on_batch(X_batch)
        err_sum += np.sum((y_batch - y_pred) ** 2)
        err_count += len(y_pred)

    mse = err_sum / err_count
    return [mse, np.sqrt(mse)]






def generate_arrays_from_file_v3(labels_lrc, index_values_lrc, image_path_base_lrc, batch_size):
    batch_features = np.zeros((batch_size, 240, 320, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        direction = random.randint(0, 2)
        labels = labels_lrc[direction]
        index_values = index_values_lrc[direction]
        image_path_base = image_path_base_lrc[direction]
        for i in range(batch_size):
            idx = np.random.choice(len(labels), 1)
            y = labels[idx]
            image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx])))
            image = np.load(image_path)
            flip_bit = random.randint(0, 1)
            if flip_bit == 1:
                image = np.flip(image, 1)
                y=y*-1
            #noise = np.random.normal(0, 25, image.shape[0]*image.shape[1]*image.shape[2]).reshape(image.shape)
            image = scipy.ndimage.interpolation.rotate(image, random.uniform(0, 360), reshape=False)
            rows, cols, chan = image.shape
            M = np.float32([[1, 0, random.randint(-50, 50)], [0, 1, random.randint(-50, 50)]])
            R = random.uniform(.95, 1.05)
            G = random.uniform(.95, 1.05)
            B = random.uniform(.95, 1.05)
            #image = np.float_(image)
            #image[:, :, 0] *= R
            #image[:, :, 1] *= G
            #image[:, :, 2] *= B
            image = cv2.warpAffine(image, M, (cols, rows))
            batch_features[i,:] = image
            batch_labels[i] = y
        yield batch_features, batch_labels




#https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image, steer, trans_range):
    rows, cols, chan = image.shape
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def generate_arrays_from_file_v4(labels_lrc, index_values_lrc, image_path_base_lrc, batch_size):
    batch_features = np.zeros((batch_size, 240, 320, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        direction = random.randint(0, 2)
        #direction = 0
        labels = labels_lrc[direction]
        index_values = index_values_lrc[direction]
        image_path_base = image_path_base_lrc[direction]
        add_amount = 0
        if direction == 1:
            add_amount=.25
        elif direction == 2:
            add_amount= -.25
        for i in range(batch_size):
            leave_loop = False
            while leave_loop==False:
                idx = np.random.choice(len(labels), 1)
                y = labels[idx]
                if abs(y) < 0.15:
                    leave_prob = np.random.uniform()
                    if leave_prob > 0.9:
                        leave_loop = True
                else:
                    leave_loop = True
            y=y+add_amount
            image_path = os.path.join(image_path_base, "{}.jpg.npy".format(int(index_values[idx])))
            image = np.load(image_path)
            flip_bit = random.randint(0, 1)
            if flip_bit == 1:
                image = np.flip(image, 1)
                y=y*-1
            image, y = trans_image(image, y, 150)
            image = add_random_shadow(image)
            image = augment_brightness_camera_images(image)
            batch_features[i,:] = image
            batch_labels[i] = y
        yield batch_features, batch_labels
