import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.base import BaseEstimator, TransformerMixin

from base import Video


class PerseptronVideoInput(BaseEstimator, TransformerMixin):
    def fit(self, video: Video, y=None):
        return self

    def transform(self, video: Video):
        a, b, c = video.imshape
        frames = np.array(video.to_array(), dtype=np.uint8)
        n = len(frames)
        return frames.reshape((n, a * b * c)) / 255


class Perseptron(BaseEstimator, TransformerMixin):
    def __init__(self, input: int = None, hiddens: list = None, epochs=50, batch_size=32, model=None):
        super().__init__()
        if model is None:
            activ_str = 'sigmoid'
            self.model = Sequential()
            self.model.add(Dense(hiddens[0], input_shape=(input,), kernel_initializer='random_uniform'))
            self.model.add(Activation(activ_str))

            if len(hiddens) > 1:
                for hidden in hiddens[1:]:
                    self.model.add(Dense(hidden, kernel_initializer='random_uniform'))
                    self.model.add(Activation(activ_str))

            self.model.add(Dense(input))
            self.model.add(Activation('softmax'))
            self.model.compile(optimizer='rmsprop',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
            self.model.summary()

        else:
            self.model = model
            self.model.summary()

        self.__epochs = epochs
        self.__batch_size = batch_size

    def fit(self, frames, y=None):
        self.model.fit(frames, frames, epochs=self.__epochs)
        return self

    def transform(self, frames):
        return self.model.predict(frames)


class PerseptronVideoOutput(BaseEstimator, TransformerMixin):
    def __init__(self, file, shape):
        super().__init__()
        self.__file = file
        self.__shape = shape

    def fit(self, X, y):
        return self

    def transform(self, frames):
        (frame.reshape(self.__shape) for frame in frames)
        return Video(self.__file, frames=frames)
