import tensorflow.keras.preprocessing.image  as preprocess_image
import numpy as np
import os
from multiprocessing.dummy import Pool
import tensorflow as tf


def load(x: list) -> np.array:
    images = []
    target_size = x[2:]
    color_mode = x[1]
    paths = x[0]
    for i in paths:
        try:
            image = preprocess_image.img_to_array(preprocess_image.load_img(i, color_mode=color_mode,
                                                                            target_size=target_size),
                                                  data_format='channels_last', dtype=np.float32) / 255
            images.append(image)
        except:
            pass
    return np.array(images)


def load_n_samples(abs_path, dim=None, color_mode='rgb', samples=100, workers=1, step=10) -> np.array:
    if not dim:
        dim = (128, 128)
    height, width = dim

    files = os.listdir(abs_path)
    files = np.array(list(map(lambda x: os.path.join(abs_path, x), files)))
    files = files[np.random.randint(0, len(files), size=samples)]
    tup_load = [(files[i:i + step], color_mode, height, width) for i in
                range(0, len(files), step)]
    with Pool(workers) as p:
        data = p.map(load, tup_load)
    return np.array([z for i in data for z in i])


class AEDatagenerator(tf.keras.utils.Sequence):
    def __init__(self, pathes, batch_size=32, dim=(32, 32), n_channels=1,
                 shuffle=True, with_noise=False, std=0.5):
        self.std = std
        self.dim = dim
        self.batch_size = batch_size
        self.pathes = pathes
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.with_noise = with_noise
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.pathes) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        pathes_temp = [self.pathes[k] for k in indexes]
        X = self.__data_generation(pathes_temp)
        if self.with_noise:
            noisy_X = self.add_noise(X)
            return noisy_X, X
        else:
            return X, X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pathes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def add_noise(self,X):
        noise = np.random.normal(0, self.std, X.shape)
        return X + noise


    def __data_generation(self, pathes_tmp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, path in enumerate(pathes_tmp):
            X[i, :] = preprocess_image.img_to_array(preprocess_image.load_img(path, color_mode='rgb',
                                                                              target_size=self.dim),
                                                    data_format='channels_last', dtype=np.float32) / 255

        return X
