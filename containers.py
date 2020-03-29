from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from backend import *
from base import *


class LinearizerVideoInput(BasePassVideoLayers):

    def transform(self, video: Video):
        a, b, c = video.imshape
        frames = np.array(video.to_array(), dtype=np.uint8)
        n = len(frames)
        return frames.reshape((n, a * b * c)) / 255


class PerseptronLayer(BaseKerasNeuralLayer):
    file_name = 'keras_model.h5'

    def build(self, *args, **kwargs):
        compress = self.context_params.compress
        video = self.context_params.video
        h, w, _ = video.imshape

        input_len = h*w
        hidden_len = int(input_len / compress)

        model = mdls.Sequential()

        model.add(lrs.Dense(hidden_len, input_shape=(input_len, )))
        model.add(lrs.Activation('sigmoid'))

        model.add(lrs.Dense(hidden_len))
        model.add(lrs.Activation('sigmoid'))

        model.add(lrs.Dense(input_len))
        model.add(lrs.Activation('sigmoid'))

        model.compile(
            optimizer='rmsprop',
            loss=losses.binary_crossentropy,
            metrics=[metrics.binary_accuracy]
        )

        return model


class LinearizerVideoOutput(BasePassVideoLayers):

    def transform(self, frames):
        video = self.context_params.video
        imshape = video.imshape

        (frame.reshape(imshape) for frame in frames)
        return Video(self.context_params.out_file_path, frames=frames)



