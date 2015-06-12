import lasagne

from .. import core


class GlorotUniformInit(core.WeightInit):

    def __init__(self, gain=1.0):
        self.gain = gain

    def initialize_value(self, var):
        init = lasagne.init.GlorotUniform(gain=self.gain)
        return init.sample(var.shape).astype(var.dtype)
