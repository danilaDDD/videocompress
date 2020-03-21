from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

import numpy as np
from matplotlib import pyplot as plt

from base import Video
from models import PerseptronModel


def keras_test():
    input_shape = 10
    output_shape = 10

    model = Sequential([
        Dense(10, input_shape=(input_shape,)),
        Activation('linear'),

        Dense(5),
        Activation('relu'),
        Dense(5),
        Activation('relu'),

        Dense(output_shape),
        Activation('linear')
    ])

    model.summary()
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    X = np.random.normal(0, 1, (100, 10))
    Y = np.random.normal(0, 10, (100, 10))
    history = model.fit(X, Y, epochs=100)

    # loss_fig = plt.figure('train')
    # plt.plot(history.history['loss'])
    # plt.title('loss')
    #
    # acc_fig = plt.figure('train')
    # plt.plot(history.history['accuracy'])
    # plt.title('acc')
    #
    # plt.show()


def simple_video_test(file_name='test.avi'):
    video = Video.get_test_video()
    model = PerseptronModel(video)
    model.fit()


if '__name__' == '__main__':
    simple_video_test()
