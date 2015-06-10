import numpy as np

from .. import core

# for importing
from ..core.inits import (SharedInit,
                          WeightInit,
                          ConstantInit,
                          ZeroInit,
                          PreallocatedInit)


class NormalWeightInit(core.WeightInit):

    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def initialize_value(self, vw):
        raw = np.random.randn(*vw.shape)
        scaled = (self.mean + self.std * raw)
        return scaled.astype(vw.dtype)
