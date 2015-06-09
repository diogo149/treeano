import numpy as np

import treeano


class OnesInitialization(treeano.SharedInitialization):

    def initialize_value(self, var):
        return np.ones(var.shape).astype(var.dtype)
