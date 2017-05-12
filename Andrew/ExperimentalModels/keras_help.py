import keras
import numpy as np
import os
import keras.backend as K


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


















