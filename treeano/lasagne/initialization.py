import lasagne

from .. import core


class GlorotUniform(core.WeightInitialization):

    # FIXME add parameters to constructor (eg. initialization_gain)

    def initialize_value(self, var):
        return lasagne.init.GlorotUniform().sample(var.shape).astype(var.dtype)
