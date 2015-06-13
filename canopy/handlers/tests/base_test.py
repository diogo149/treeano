import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_handled_function():
    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_function(network, [], ["i"], ["i"])
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x, fn(x)[0])


def test_call_handler1():
    class x_equals_3(canopy.handlers.NetworkHandlerImpl):

        def call(self, fn, x, *args, **kwargs):
            if x == 3:
                return fn(*args, **kwargs)
            else:
                assert False

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_function(network,
                                          [x_equals_3()],
                                          ["i"],
                                          ["i"])
    y = np.array(3, dtype=fX)
    np.testing.assert_equal(y, fn(3, y)[0])

    @nt.raises(AssertionError)
    def tmp():
        fn(2, y)

    tmp()


def test_call_handler2():
    class x_equals_3(canopy.handlers.NetworkHandlerImpl):

        def call(self, fn, x, *args, **kwargs):
            if x == 3:
                return fn(*args, **kwargs)
            else:
                assert False

    class plus_n(canopy.handlers.NetworkHandlerImpl):

        def __init__(self, n):
            self.n = n

        def call(self, fn, *args, **kwargs):
            res = fn(*args, **kwargs)
            res[0] += self.n
            return res

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_function(network,
                                          [plus_n(42), x_equals_3()],
                                          ["i"],
                                          ["i"])
    y = np.array(3, dtype=fX)
    np.testing.assert_equal(y + 42, fn(3, y)[0])

    @nt.raises(AssertionError)
    def tmp():
        fn(2, y)

    tmp()


def test_call_handler3():
    class x_equals_3(canopy.handlers.NetworkHandlerImpl):

        def call(self, fn, x, *args, **kwargs):
            if x == 3:
                return fn(*args, **kwargs)
            else:
                assert False

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_function(network,
                                          [x_equals_3(), x_equals_3()],
                                          ["i"],
                                          ["i"])
    y = np.array(3, dtype=fX)
    np.testing.assert_equal(y, fn(3, 3, y)[0])

    @nt.raises(AssertionError)
    def tmp():
        fn(2, y)

    tmp()


def test_build_handler1():
    class network_identity(canopy.handlers.NetworkHandlerImpl):

        def transform_network(self, network):
            return network

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_function(network,
                                          [network_identity()],
                                          ["i"],
                                          ["i"])
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x, fn(x)[0])


def test_build_handler2():
    class new_network(canopy.handlers.NetworkHandlerImpl):

        def transform_network(self, network):
            return tn.InputNode("new_node", shape=()).network()

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_function(network,
                                          [new_network()],
                                          ["new_node"],
                                          ["new_node"])
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x, fn(x)[0])


def test_build_handler3():
    class add_42(canopy.handlers.NetworkHandlerImpl):

        def transform_network(self, network):
            return tn.SequentialNode(
                "seq",
                [network.root_node,
                 tn.toy.AddConstantNode("ac", value=42)]
            ).network()

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_function(network,
                                          [add_42()],
                                          ["i"],
                                          ["ac"])
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x + 42, fn(x)[0])
