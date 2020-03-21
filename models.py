from os.path import join
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model
import cv2
import os

from containers import *
from params import *


class PerseptronModel:
    @staticmethod
    def load_model():
        saved_dir = join(MPATH, PerseptronModel.name)
        keras_model = load_model('keras_model.h5')
        add_file = open(join(saved_dir, 'add.txt'), 'r')
        height, weight, deep, epochs = add_file.read().split(',')
        line_size = height*weight*deep

        return Pipeline([
            ('video_input', PerseptronVideoInput()),
            ('perseptron', Perseptron(line_size, [int(line_size / 2)], epochs=epochs, model=keras_model)),
            ('video_output', PerseptronVideoOutput((height, weight, deep)))
        ])

    name = 'perseptron'

    def __init__(self, video, epochs=20, model=None):
        height, width, deep = video.imshape
        line_size = height * width * deep

        if model is None:
            model = Pipeline([
                ('video_input', PerseptronVideoInput()),
                ('perseptron', Perseptron(line_size, [int(line_size / 2)], epochs=epochs)),
                ('video_output', PerseptronVideoOutput(video.file, (height, width, deep)))
            ])

        self.model = model
        self.video = video
        self.epochs = epochs

    def fit(self):
        frames = self.video.to_array()
        self.model = self.model.fit(self.video, None)


def test(self):
    count = 0

    for frame in self.__frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = self.model.transform(gray_frame)
        cv2.imshow("result", gray_frame)
        count += 1

    cv2.release()
    cv2.destroyAllWindows()


def save(self, index=''):
    saved_dir = join(MPATH, self.name)
    try:
        os.mkdir(saved_dir)
    except OSError:
        pass

    keras_model = self.model.get('perseptron_model')
    keras_model.save(join(saved_dir, 'keras_model.h5'))

    add_file = open(join(saved_dir, 'add.txt'), 'w')
    add_file.write(self.video)
    add_file.close()
