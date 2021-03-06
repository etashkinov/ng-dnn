import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os import path
from os import environ
import tensorflow as tf
import numpy as np

STORAGE_PATH = environ.get("STORAGE_PATH")
MODEL_PATH = STORAGE_PATH + "/model/model"
LABELS = ('circle', 'line', 'arch')


def create_model():
    network = input_data(shape=[None, 28, 28, 1], name='input', dtype=tf.float32)
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, len(LABELS), activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')
    return tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir='board')


def img_to_array(image):
    result = image.convert('RGB').convert('L').resize((28, 28), 1)
    result = np.asarray(result)
    x = []
    for row in result:
        we = []
        x.append(we)
        for value in row:
            we.append(1 - value / 255)
    result = np.array(x).reshape(28, 28, 1)

    return result


def get_one_hot(label):
    result = [0] * len(LABELS)
    result[LABELS.index(label)] = 1
    return result


def split(data):
    x = []
    y = []
    for d in data:
        x.append(img_to_array(d[0]).tolist())
        y.append(get_one_hot(d[1]))

    return np.array(x), np.array(y)


def load(model):
    if path.isfile(MODEL_PATH + ".index"):
        model.load(MODEL_PATH)
        print('Model loaded from', MODEL_PATH)
        return True
    else:
        print('Model not found at', MODEL_PATH)
        return False


def fit(model, data):
    np.random.shuffle(data)
    x, y = split(data)
    model.fit({'input': x}, {'target': y}, n_epoch=60, show_metric=True)
    model.save(MODEL_PATH)
    print('Model fitted and saved:', MODEL_PATH)


class FigureNode:
    def __init__(self) -> None:
        super().__init__()

        self.model = create_model()

    def load(self):
        return load(self.model)

    def fit(self, data):
        with tf.Graph().as_default():
            fit_model = create_model()
            load(fit_model)
            fit(fit_model, data)
            self.model = fit_model

    def pre_fit(self, data):
        fit(self.model, data)

    def predict(self, image):
        array = img_to_array(image)
        prediction = self.model.predict((array,))
        result = dict(zip(LABELS, prediction[0]))
        return result
