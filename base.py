from abc import ABC
from os.path import join
import time
from collections import deque

import cv2
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

from params import VPATH, MPATH


class Video(cv2.VideoCapture):
    codec = cv2.VideoWriter_fourcc(*'XVID')

    @staticmethod
    def get_test_video(frame_shape=(10, 10), size=10, file='test.avi'):
        frames = np.random.randint(0, 255, (*frame_shape, size))
        source = join(VPATH, file)
        return Video(source, frames=frames)

    @staticmethod
    def __is_pressed_key(key):
        return cv2.waitKey(1) & 0xFF == ord(key)

    def __init__(self, file, skip=1, fps=50, frames=None, codec=None):
        if codec is not None:
            self.codec = codec

        if frames is not None:
            first_frame = np.uint8(next(frames))
            imshape = first_frame.shape
            a, b, _ = imshape
            if a != b:
                raise IOError('width != height frame!')

            out = cv2.VideoWriter(file, Video.codec, fps, imshape[:2])
            out.write(first_frame)

            for frame in frames:
                out.write(np.uint8(frame))

            out.release()

        super().__init__(file)
        self.fps = fps
        self.file = file
        self.skip = skip
        frame = self.get_frame(0)
        if frame is not None:
            self.__imshape = frame.shape
        else:
            raise IOError('frames[0]==None!')

    @property
    def imshape(self):
        return self.__imshape

    @property
    def fps(self):
        return self.__fps

    @fps.setter
    def fps(self, value):
        self.set(cv2.CAP_PROP_FPS, value)

    def track(self, skip=None):
        if skip is not None:
            self.skip = skip

        count = 0
        while self.isOpened():
            self.set(cv2.CAP_PROP_POS_FRAMES, count * self.skip)
            grab, frame = self.read()
            if grab:
                yield frame
            else:
                break

            if self.__is_pressed_key('q'):
                break

            count += 1

        else:
            raise IOError('not open!')
        self.release()

    def show(self, delay=0.1):
        print('show')
        count = 0
        if self.isOpened():
            for frame in self.track():
                cv2.imshow('frame', frame)
                time.sleep(delay)
                if self.__is_pressed_key('q'):
                    break
                print(count)
                count += 1
        else:
            raise IOError('not open')
        self.release()
        cv2.destroyAllWindows()

    def get_frame(self, idx):
        if self.isOpened():
            self.set(cv2.CAP_PROP_POS_FRAMES, idx * self.skip)
            grab, frame = self.read()
            if grab:
                return frame
            else:
                return None
        else:
            raise IOError('not open')

    def reopen(self, file=None):
        if file is None:
            file = self.file
        else:
            self.file = file

        self.open(file)
        return self.isOpened()

    def to_array(self):
        try:
            idx = 0
            frames = deque()
            while True:
                frame = self.get_frame(idx)

                if self.__is_pressed_key('q'):
                    break

                if frame is None:
                    break

                frames.append(frame)
                idx += 1

            return np.array(frames, dtype=np.uint8)
        except IOError as e:
            raise e

    def __str__(self):
        a, b, c = self.imshape
        return "f{a};{b};{c}"


class BaseSkLayer(BaseEstimator, TransformerMixin):
    """
        implement
        @transform
        @file_name
        @save
    """

    file_name = None
    context_params = None

    def __init__(self, context_params=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_params = context_params

    def fit(self, *args):
        return self

    def transform(self, *args):
        return None

    def save(self, *args):
       pass

    def load(self, *args):
        pass


class BasePassVideoLayers(BaseSkLayer):

    def fit(self, video: Video) -> BaseSkLayer:
        return self

    def transform(self, video: Video) -> np.ndarray:
        raise NotImplementedError('not implemented method')


class BaseKerasNeuralLayer(BaseSkLayer):
    """
    implements
    @build
    @load
    """

    def build(self, *args, **kwargs):
        raise NotImplementedError('not implemented build')

    def load(self, *args, **kwargs):
        raise NotImplementedError('not implemented build')

    def __init__(self, context=None, *args, **kwargs):
        super().__init__(context)

        if context is None:
            self.model = self.build(*args, **kwargs)

        else:
            self.model = self.load(*args, **kwargs)

        self.model.summary()

    def transform(self, X):
        if self.model is None:
            raise NotImplementedError('model not found')
        else:
            return self.model.predict(X)

    def fit(self, frames, epochs, *args):
        self.model.fit(frames, frames, epochs=epochs)
        return self

    def save(self, *args):
        raise NotImplementedError('save layer not implemented')

    def load(self, *args):
        raise NotImplementedError('save layer not implemented')


class BaseVideoModel:
    name = None

    def __init__(self, video: Video, compress=2, is_scala=True, out_file_path=None, *args, **kwargs):
        self.__video = video
        self.__compress = compress
        self.__is_scala = is_scala
        self.__epochs = None
        self.__saved_dir = join(MPATH, self.name)

        if out_file_path is None:
            self.__out_file_path = video.file
        else:
            self.__out_file_path = out_file_path


    def fit(self, epochs=20, *args):
        self.__epochs = epochs
        return self

    @property
    def video(self):
        return self.__video

    @property
    def compress(self):
        return self.__compress

    @property
    def is_scalar(self):
        return self.__is_scala

    @property
    def epochs(self):
        return self.__epochs

    @property
    def saved_dir(self):
        return self.__saved_dir

    @property
    def out_file_name(self):
        return self.__out_file_path


class BasePipeVideoModel(BaseVideoModel):
    """implements:
    @InputLayerClass
    @NeuralLayerClass
    @OutputLayerClass
    @name
    """

    InputLayerClass = None
    NeuralLayerClass = None
    OutputLayerClass = None

    def __init__(self, video: Video, compress=2, is_scala=True, *args, **kwargs):
        """Create @pipe_model by @self=context and load"""
        super().__init__(video, compress, is_scala, *args, **kwargs)

        if video:
            self.pipe_model = self.build(load=False)
        else:
            self.pipe_model = self.build(load=True)


    def fit(self, epochs=20, *args):
        self.pipe_model.fit(self.video, epochs)
        return super().fit(epochs, *args)

    def save(self):
        for layer in self.pipe_model:
            layer.save(self.name)

    def fit_transform(self, video: Video, *args):
        return self.pipe_model.fit_transform(video, *args)

    def build(self, load=False):
        if load:
            context = None
        else:
            context = self

        self.pipe_model = Pipeline([
            ('video_input', self.InputLayerClass(context)),
            ('network', self.NeuralLayerClass(context)),
            ('output_video', self.OutputLayerClass(context))
        ])






