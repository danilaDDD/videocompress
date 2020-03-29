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
        h, w = video.imshape

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
            optimizer=optimizers.RMSprop,
            loss=losses.binary_crossentropy,
            metrics=[metrics.binary_accuracy]
        )

        return model

    def save(self, *args):
        saved_dir = self.context_params.saved_dir
        self.saved_path = join(saved_dir, self.file_name)
        self.model.save(self.saved_path)

    def load(self, *args, **kwargs):
        return mdls.load_model(self.saved_path)


class LinearizerVideoOutput(BasePassVideoLayers):

    def transform(self, frames):
        video = self.context_params.video
        imshape = video.imshape

        (frame.reshape(imshape) for frame in frames)
        return Video(self.context_params.out_file_path, frames=frames)



