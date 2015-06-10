import lasagne

from .. import core


class GlorotUniformInit(core.WeightInitialization):

    # FIXME add parameters to constructor (eg. initialization_gain)

    def initialize_value(self, var):
        return lasagne.init.GlorotUniform().sample(var.shape).astype(var.dtype)
