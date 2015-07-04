import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_handled_fn():
    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(network, [], {"x": "i"}, {"out": "i"})
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x, fn({"x": x})["out"])


def test_call_handler1():
    class y_equals_3(canopy.handlers.NetworkHandlerImpl):

        def call(self, fn, in_dict, *args, **kwargs):
            if in_dict["y"] == 3:
                return fn(in_dict, *args, **kwargs)
            else:
                assert False

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(network,
                                    [y_equals_3()],
                                    {"x": "i"},
                                    {"out": "i"})
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x, fn({"x": x, "y": 3})["out"])

    @nt.raises(AssertionError)
    def tmp():
        fn({"x": x, "y": 2})

    tmp()


def test_call_handler2():
    class y_equals_3(canopy.handlers.NetworkHandlerImpl):

        def call(self, fn, in_dict, *args, **kwargs):
            if in_dict["y"] == 3:
                return fn(in_dict, *args, **kwargs)
            else:
                assert False

    class plus_n(canopy.handlers.NetworkHandlerImpl):

        def __init__(self, n):
            self.n = n

        def call(self, fn, *args, **kwargs):
            res = fn(*args, **kwargs)
            res["out"] += self.n
            return res

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(network,
                                    [plus_n(42), y_equals_3()],
                                    {"x": "i"},
                                    {"out": "i"})
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x + 42, fn({"x": x, "y": 3})["out"])

    @nt.raises(AssertionError)
    def tmp():
        fn({"x": x, "y": 2})

    tmp()


def test_build_handler1():
    class network_identity(canopy.handlers.NetworkHandlerImpl):

        def transform_network(self, network):
            return network

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(network,
                                    [network_identity()],
                                    {"x": "i"},
                                    {"out": "i"})
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x, fn({"x": x})["out"])


def test_build_handler2():
    class new_network(canopy.handlers.NetworkHandlerImpl):

        def transform_network(self, network):
            return tn.InputNode("new_node", shape=()).network()

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(network,
                                    [new_network()],
                                    {"x": "new_node"},
                                    {"out": "new_node"})
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x, fn({"x": x})["out"])


def test_build_handler3():
    class add_42(canopy.handlers.NetworkHandlerImpl):

        def transform_network(self, network):
            return tn.SequentialNode(
                "seq",
                [network.root_node,
                 tn.AddConstantNode("ac", value=42)]
            ).network()

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(network,
                                    [add_42()],
                                    {"x": "i"},
                                    {"out": "ac"})
    x = np.array(3, dtype=fX)
    np.testing.assert_equal(x + 42, fn({"x": x})["out"])
