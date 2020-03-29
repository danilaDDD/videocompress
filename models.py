from containers import *
from base import BasePipeVideoModel


class PerseptronModel(BasePipeVideoModel):
    name = 'perseptron'

    InputLayerClass = LinearizerVideoInput
    NeuralLayerClass = PerseptronLayer
    OutputLayerClass = LinearizerVideoOutput



