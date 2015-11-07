from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import toolz
import numpy as np
import theano
import theano.tensor as T

from .. import utils
from .inits import ZeroInit

ENABLE_TEST_VALUE = theano.config.compute_test_value != "off"

VALID_TAGS = set("""
input
output
weight
bias
parameter
hyperparameter
monitor
state
tied
""".split())


class VariableWrapper(object):

    def __init__(self,
                 name,
                 shape=None,
                 dtype=None,
                 broadcastable=None,
                 is_shared=None,
                 tags=None,
                 ndim=None,
                 variable=None,
                 inits=None,
                 relative_network=None,
                 default_inits=None,
                 default_inits_hyperparameters=("inits",)):
        self.name = name
        self.shape_ = shape
        self.dtype_ = dtype
        self.broadcastable_ = broadcastable
        self.is_shared_ = is_shared
        self.tags_ = tags
        self.ndim_ = ndim
        self.variable_ = variable
        if default_inits is not None:
            # replace inits with calculated inits from network
            assert inits is None
            assert relative_network is not None
            assert isinstance(default_inits, (list, tuple))
            inits = list(toolz.concat(relative_network.find_hyperparameters(
                default_inits_hyperparameters,
                default_inits)))
        self.inits = inits
        # relative_network is provided so that variables can auto-compute
        # their shape
        self.relative_network = relative_network
        self.validate()

    def to_state(self, name):
        return dict(
            shape=self.shape_,
            dtype=self.dtype_,
            broadcastable=self.broadcastable_,
            is_shared=self.is_shared_,

        )

    def validate(self):
        shape = self.shape_
        dtype = self.dtype_
        broadcastable = self.broadcastable_
        is_shared = self.is_shared_
        tags = self.tags_
        ndim = self.ndim_
        variable = self.variable_

        if ndim is not None and shape is not None:
            assert len(shape) == ndim
        if ndim is not None and variable is not None:
            assert ndim == variable.ndim
        if broadcastable is not None and variable is not None:
            assert broadcastable == variable.broadcastable
        if is_shared is not None and variable is not None:
            assert is_shared == utils.is_shared_variable(self.variable)
        if dtype is not None and variable is not None:
            assert dtype == variable.dtype
        if tags is not None:
            self.verify_tags(set(tags))
        if is_shared:

            assert self.inits is not None, dict(
                name=self.name,
                msg=("if inits is None, then this shared variable cannot be "
                     "shared or loaded"),
            )

    def verify_tags(self, tags):
        for tag in tags:
            assert tag in VALID_TAGS
        if self.is_shared:
            # exactly one of parameter and state should be set
            assert ("parameter" in tags) != ("state" in tags)
            # the only valid tags for shared are the following:
            assert len(tags - {"weight", "bias", "parameter", "state"}) == 0
        else:
            assert len({"weight", "bias", "parameter", "state"} & tags) == 0

    @property
    def is_shared(self):
        if self.is_shared_ is None:
            # if is_shared is not supplied, a variable must be supplied
            assert self.variable_ is not None
            self.is_shared_ = utils.is_shared_variable(self.variable)
        return self.is_shared_

    @property
    def tags(self):
        if self.tags_ is None:
            self.tags_ = []
        if not isinstance(self.tags_, set):
            self.tags_ = set(self.tags_)
        self.verify_tags(self.tags_)
        return self.tags_

    @property
    def ndim(self):
        if self.ndim_ is None:
            if self.variable_ is not None:
                self.ndim_ = self.variable_.ndim
            elif self.shape_ is not None:
                self.ndim_ = len(self.shape_)
            else:
                raise ValueError("ndim not defined")
        return self.ndim_

    @property
    def dtype(self):
        if self.dtype_ is None:
            if self.variable_ is not None:
                self.dtype_ = self.variable_.dtype
            else:
                self.dtype_ = theano.config.floatX
        return self.dtype_

    @property
    def broadcastable(self):
        if self.broadcastable_ is None:
            if self.variable_ is not None:
                self.broadcastable_ = self.variable_.broadcastable
            else:
                self.broadcastable_ = (False, ) * self.ndim
        return self.broadcastable_

    @property
    def variable(self):
        if self.variable_ is None:
            if self.is_shared:
                # find appropriate initialization scheme
                inits = self.inits
                if inits is None:
                    inits = []
                for initialization in inits:
                    if initialization.predicate(self):
                        break
                else:
                    # default to zero initialization if none work
                    initialization = ZeroInit()

                # create the shared variable
                variable = initialization.create_shared(self)
            else:
                variable = T.TensorType(self.dtype,
                                        self.broadcastable)(self.name)
            self.variable_ = variable

            # for ease of debugging, add test values
            # ---
            # this must be done after self.variable_ is set to avoid a
            # recursive loop when calling self.shape
            if (not self.is_shared) and ENABLE_TEST_VALUE:
                test_value = np.random.rand(*self.shape).astype(self.dtype)
                variable.tag.test_value = test_value

            # re-validate that initialization worked properly
            self.validate()

        return self.variable_

    @property
    def shape(self):
        if self.shape_ is None:
            # cannot derive shape for shared variable
            assert not self.is_shared
            # TODO
            # can we call variable.get_value().shape
            # we don't know if one of the intended dimensions is None,
            # but it seems like a fair assumption that a shared variable
            # won't change shape (maybe we can add a flag of whether or not
            # shape doesn't change that defaults to True)

            # FIXME calculate shape
            # assert False
            self.shape_ = tuple(self.variable_.shape.eval())
        return self.shape_

    def symbolic_shape(self):
        """
        returns the shape of the variable with None's replaced by symbolic
        variables
        """
        shape = self.shape
        var_shape = self.variable.shape
        out_shape = []
        for i, s in enumerate(shape):
            if s is not None:
                out_shape.append(s)
            else:
                out_shape.append(var_shape[i])
        return tuple(out_shape)

    @property
    def value(self):
        assert self.is_shared
        return self.variable.get_value()

    @value.setter
    def value(self, new_value):
        assert new_value.dtype == self.dtype
        assert new_value.shape == self.shape
        self.variable.set_value(new_value)

    def __repr__(self):
        return "{cls}(name={name})".format(cls=self.__class__.__name__,
                                           name=self.name)
