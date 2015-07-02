import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_function_combine_node_serialization():
    tn.check_serialization(tn.InputFunctionCombineNode("a"))


def test_concatenate_node_serialization():
    tn.check_serialization(tn.ConcatenateNode("a", []))
    tn.check_serialization(tn.ConcatenateNode(
        "a",
        [tn.ConcatenateNode("b", []),
         tn.ConcatenateNode("c", [])]))


def test_elementwise_sum_node_serialization():
    tn.check_serialization(tn.ElementwiseSumNode("a", []))
    tn.check_serialization(tn.ElementwiseSumNode(
        "a",
        [tn.ElementwiseSumNode("b", []),
         tn.ElementwiseSumNode("c", [])]))


def test_input_elementwise_sum_node_serialization():
    tn.check_serialization(tn.InputElementwiseSumNode("a"))


def test_elementwise_product_node_serialization():
    tn.check_serialization(tn.ElementwiseProductNode("a", []))
    tn.check_serialization(tn.ElementwiseProductNode(
        "a",
        [tn.ElementwiseProductNode("b", []),
         tn.ElementwiseProductNode("c", [])]))


def test_input_function_combine_node():
    def fcn_network(combine_fn):
        network = tn.ContainerNode("c", [
            tn.SequentialNode(
                "s1",
                [tn.InputNode("in1", shape=(3, 4, 5)),
                 tn.SendToNode("stn1", reference="fcn", to_key="b")]),
            tn.SequentialNode(
                "s2",
                [tn.InputNode("in2", shape=(3, 4, 5)),
                 tn.SendToNode("stn2", reference="fcn", to_key="a")]),
            tn.InputFunctionCombineNode("fcn", combine_fn=combine_fn)
        ]).network()
        return network.function(["in1", "in2"], ["fcn"])
    x = np.random.randn(3, 4, 5).astype(fX)
    y = np.random.randn(3, 4, 5).astype(fX)
    fn1 = fcn_network(lambda *args: sum(args))
    np.testing.assert_allclose(fn1(x, y)[0], x + y)
    fn2 = fcn_network(lambda x, y: x * y)
    np.testing.assert_allclose(fn2(x, y)[0], x * y)
    # testing alphabetical ordering of to_key
    # ---
    # adding other key times 0 to avoid unused input error
    fn3 = fcn_network(lambda x, y: x + 0 * y)
    np.testing.assert_allclose(fn3(x, y)[0], y)
    fn4 = fcn_network(lambda x, y: y + 0 * x)
    np.testing.assert_allclose(fn4(x, y)[0], x)


def test_concatenate_node():

    def replace_nones(shape):
        return [s if s is not None else np.random.randint(1, 100)
                for s in shape]

    for axis, s1, s2, s3 in [[0] + [(3, 2, 4)] * 3,
                             [0, (3, 2, 4), (5, 2, 4), (6, 2, 4)],
                             [0, (3, 2, 4), (5, 2, 4), (None, 2, 4)],
                             [1, (3, 5, 4), (3, 1, 4), (3, 2, 4)], ]:
        network = tn.ConcatenateNode(
            "concat",
            [tn.InputNode("i1", shape=s1),
             tn.InputNode("i2", shape=s2),
             tn.InputNode("i3", shape=s3)],
            axis=axis,
        ).network()
        fn = network.function(["i1", "i2", "i3"], ["concat"])
        i1 = np.random.rand(*replace_nones(s1)).astype(fX)
        i2 = np.random.rand(*replace_nones(s2)).astype(fX)
        i3 = np.random.rand(*replace_nones(s3)).astype(fX)
        np.testing.assert_allclose(np.concatenate([i1, i2, i3], axis=axis),
                                   fn(i1, i2, i3)[0])


def test_concatenate_node_wrong_shape():

    @nt.raises(AssertionError)
    def build_concatenate(axis, s1, s2, s3):
        tn.ConcatenateNode(
            "concat",
            [tn.InputNode("i1", shape=s1),
             tn.InputNode("i2", shape=s2),
             tn.InputNode("i3", shape=s3)],
            axis=axis,
        ).network().build()

    for args in [[100] + [(3, 2, 4)] * 3,
                 [0, (3, 2, 4), (3, 2, 4), (3, 3, 4)],
                 [0, (3, 2, 4), (3, 2, 4), (3, None, 4)], ]:
        build_concatenate(*args)


def test_elementwise_sum_node():
    for s in [(),
              (3, 4, 5)]:
        network = tn.ElementwiseSumNode(
            "es",
            [tn.InputNode("i1", shape=s),
             tn.InputNode("i2", shape=s),
             tn.InputNode("i3", shape=s)],
        ).network()
        fn = network.function(["i1", "i2", "i3"], ["es"])
        i1 = np.array(np.random.rand(*s), dtype=fX)
        i2 = np.array(np.random.rand(*s), dtype=fX)
        i3 = np.array(np.random.rand(*s), dtype=fX)
        np.testing.assert_allclose(i1 + i2 + i3,
                                   fn(i1, i2, i3)[0],
                                   rtol=1e-5)


def test_input_elementwise_sum_node():
    for s in [(),
              (3, 4, 5)]:
        network = tn.ContainerNode(
            "all",
            [tn.InputElementwiseSumNode("ies"),
             tn.SequentialNode(
                 "seq1",
                [tn.InputNode("i1", shape=s),
                 tn.SendToNode("st1", reference="ies", to_key="in1")]),
             tn.SequentialNode(
                 "seq2",
                 [tn.InputNode("i2", shape=s),
                  tn.SendToNode("st2", reference="ies", to_key="in2")]),
             tn.SequentialNode(
                 "seq3",
                 [tn.InputNode("i3", shape=s),
                  tn.SendToNode("st3", reference="ies", to_key="in3")])]
        ).network()
        fn = network.function(["i1", "i2", "i3"], ["ies"])
        i1 = np.array(np.random.rand(*s), dtype=fX)
        i2 = np.array(np.random.rand(*s), dtype=fX)
        i3 = np.array(np.random.rand(*s), dtype=fX)
        np.testing.assert_allclose(i1 + i2 + i3,
                                   fn(i1, i2, i3)[0],
                                   rtol=1e-5)


def test_elementwise_product_node():
    for s in [(),
              (3, 4, 5)]:
        network = tn.ElementwiseProductNode(
            "es",
            [tn.InputNode("i1", shape=s),
             tn.InputNode("i2", shape=s),
             tn.InputNode("i3", shape=s)],
        ).network()
        fn = network.function(["i1", "i2", "i3"], ["es"])
        i1 = np.array(np.random.rand(*s), dtype=fX)
        i2 = np.array(np.random.rand(*s), dtype=fX)
        i3 = np.array(np.random.rand(*s), dtype=fX)
        np.testing.assert_allclose(i1 * i2 * i3,
                                   fn(i1, i2, i3)[0],
                                   rtol=1e-5)
