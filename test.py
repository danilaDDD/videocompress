import numpy as np
import os
from matplotlib import pyplot as plt

from backend import *
from base import Video
from local import VPATH
from models import PerseptronModel


def keras_test():
    input_shape = 10
    output_shape = 10

    model = mdls.Sequential()
    model.add(lrs.Dense(10, input_shape=(input_shape,)))
    model.add(lrs.Activation('linear'))

    model.add(lrs.Dense(5))
    model.add(lrs.Activation('relu'))
    model.add(lrs.Dense(5))
    model.add(lrs.Activation('relu'))

    model.add(lrs.Dense(output_shape))
    model.add(lrs.Activation('linear'))

    model.summary()
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    X = np.random.normal(0, 1, (100, 10))
    Y = np.random.normal(0, 10, (100, 10))
    model.fit(X, Y, epochs=100)

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
    video = Video.get_test_video(frame_shape=(100, 100))
    model = PerseptronModel(video)
    model.fit()


def model_test():
    source_path = os.path.join(VPATH, 'random_video.avi')
    video = Video(source_path)

    model = PerseptronModel(video)
    model = model.fit()
    model.save()


model_test()

