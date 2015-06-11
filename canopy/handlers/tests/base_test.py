import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_handled_function():
    network = tn.InputNode("i", shape=()).build()
    fn = canopy.handlers.handled_function(network, [], ["i"], ["i"])
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x, fn(x)[0])


def test_call_handler1():
    # TODO move to decorator_test
    @canopy.handlers.call_handler
    def x_equals_3(fn, x, *args, **kwargs):
        if x == 3:
            return fn(*args, **kwargs)
        else:
            assert False

    network = tn.InputNode("i", shape=()).build()
    fn = canopy.handlers.handled_function(network, [x_equals_3], ["i"], ["i"])
    y = np.array(3, dtype=fX)
    np.testing.assert_equal(y, fn(3, y)[0])

    @nt.raises(AssertionError)
    def tmp():
        fn(2, y)

    tmp()


def test_call_handler2():
    # TODO move to decorator_test
    @canopy.handlers.call_handler
    def x_equals_3(fn, x, *args, **kwargs):
        if x == 3:
            return fn(*args, **kwargs)
        else:
            assert False

    def plus_n(n):
        @canopy.handlers.call_handler
        def inner(fn, *args, **kwargs):
            res = fn(*args, **kwargs)
            res[0] += n
            return res

        return inner

    network = tn.InputNode("i", shape=()).build()
    fn = canopy.handlers.handled_function(network, [plus_n(42),
                                                    x_equals_3], ["i"], ["i"])
    y = np.array(3, dtype=fX)
    np.testing.assert_equal(y + 42, fn(3, y)[0])

    @nt.raises(AssertionError)
    def tmp():
        fn(2, y)

    tmp()
