from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import lasagne

# ############################### base classes ###############################


class SharedInitialization(object):

    """
    interface for initialization schemes of shared variables
    """

    def predicate(self, var):
        """
        whether or not the current initialization applies to the current
        variable
        """
        return True

    def create_shared(self, var):
        """
        creates the shared variable with an appropriately initialized value
        """
        kwargs = {}
        if len(var.broadcastable) > 0:
            kwargs["broadcastable"] = var.broadcastable
        value = self.initialize_value(var)
        kwargs["name"] = var.name
        kwargs["value"] = np.array(value).astype(var.dtype)
        variable = theano.shared(**kwargs)
        return variable

    def initialize_value(self, var):
        """
        creates appropriately initialized value for the given
        LazyWrappedVariable
        """
        raise NotImplementedError


class WeightInitialization(SharedInitialization):

    """
    base class for initializations that only work on weights
    """

    def predicate(self, var):
        return "weight" in var.tags

# ############################# implementations #############################


class ExceptionInitialization(SharedInitialization):

    """
    initialization scheme that always throws an exception - so that
    initialization doesn't fall back to other schemes
    """

    def initialize_value(self, var):
        assert False, "Initialization failed"


class ZeroInitialization(SharedInitialization):

    """
    initializes shared variable to zeros
    """

    def initialize_value(self, var):
        if len(var.broadcastable) > 0:
            value = np.zeros(var.shape)
        else:
            value = 0
        return value


class PreallocatedInitialization(SharedInitialization):

    """
    uses already defined shared variables and does NOT overwrite their
    values
    """

    def __init__(self, name_to_shared):
        self.name_to_shared = name_to_shared

    def predicate(self, var):
        return var.name in self.name_to_shared

    def create_shared(self, var):
        shared = self.name_to_shared[var.name]
        assert shared.dtype == var.dtype
        assert shared.get_value().shape == var.shape
        assert shared.name == var.name
        assert shared.broadcastable == var.broadcastable
        return shared


# ######################### lasagne initializations #########################


class GlorotUniform(WeightInitialization):

    # FIXME add parameters to constructor (eg. initialization_gain)

    def initialize_value(self, var):
        return lasagne.init.GlorotUniform().sample(var.shape).astype(var.dtype)
