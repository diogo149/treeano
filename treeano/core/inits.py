from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano

from .. import utils

# ############################### base classes ###############################


class SharedInit(object):

    """
    interface for initialization schemes of shared variables
    """

    def predicate(self, vw):
        """
        whether or not the current initialization applies to the current
        variable
        """
        return True

    def create_shared(self, vw):
        """
        creates the shared variable with an appropriately initialized value
        """
        kwargs = {}
        if len(vw.broadcastable) > 0:
            kwargs["broadcastable"] = vw.broadcastable
        value = self.initialize_value(vw)
        kwargs["name"] = vw.name
        kwargs["value"] = np.array(value).astype(vw.dtype)
        variable = theano.shared(**kwargs)
        return variable

    def initialize_value(self, vw):
        """
        creates appropriately initialized value for the given
        VariableWrapper
        """
        raise NotImplementedError


class WeightInit(SharedInit):

    """
    base class for initializations that only work on weights
    """

    def predicate(self, vw):
        return "weight" in vw.tags


class LinearWeightInit(WeightInit):

    """
    base class for initializations that work on linear weights (ndim >= 2)
    """

    def predicate(self, vw):
        return super(LinearWeightInit, self).predicate(vw) and vw.ndim >= 2

# ############################# implementations #############################


class ExceptionInit(SharedInit):

    """
    initialization scheme that always throws an exception - so that
    initialization doesn't fall back to other schemes
    """

    def initialize_value(self, vw):
        assert False, "Initialization failed"


class ConstantInit(SharedInit):

    """
    initializes shared variable to a constant
    """

    def __init__(self, constant):
        self.constant = constant

    def initialize_value(self, vw):
        if utils.is_ndarray(self.constant):
            assert vw.shape == self.constant.shape
            value = self.constant
        elif vw.ndim > 0:
            value = self.constant * np.ones(vw.shape)
        else:
            value = self.constant
        return value


def ZeroInit():
    """
    initializes shared variable to zeros
    """
    return ConstantInit(0)


class PreallocatedInit(SharedInit):

    """
    uses already defined shared variables and does NOT overwrite their
    values
    """

    def __init__(self, name_to_shared):
        self.name_to_shared = name_to_shared

    def predicate(self, vw):
        return vw.name in self.name_to_shared

    def create_shared(self, vw):
        shared = self.name_to_shared[vw.name]
        assert shared.dtype == vw.dtype
        assert shared.get_value().shape == vw.shape
        assert shared.name == vw.name
        assert shared.broadcastable == vw.broadcastable
        return shared
