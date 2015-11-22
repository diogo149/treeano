from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_pool_output_shape_2d():
    def test_same(input_shape, local_sizes, strides, pads, ignore_border):
        res = tn.downsample.pool_output_shape(
            input_shape,
            (2, 3),
            local_sizes,
            strides,
            pads,
            ignore_border,
        )
        from theano.tensor.signal.downsample import max_pool_2d
        ans = max_pool_2d(
            T.constant(np.random.randn(*input_shape).astype(fX)),
            ds=local_sizes,
            st=strides,
            ignore_border=ignore_border,
            padding=pads,
        ).shape.eval()
        print(ans, res)
        np.testing.assert_equal(ans, res)

    # tests w/ ignore border
    test_same((1, 1, 5, 6), (2, 3), (1, 1), (0, 0), True)
    test_same((1, 1, 5, 6), (2, 3), (2, 2), (0, 0), True)
    test_same((1, 1, 1, 1), (2, 3), (2, 2), (0, 0), True)
    test_same((1, 1, 5, 6), (2, 3), (1, 1), (0, 1), True)
    test_same((1, 1, 5, 6), (2, 3), (2, 2), (1, 0), True)
    test_same((1, 1, 1, 1), (2, 3), (2, 2), (1, 1), True)

    # tests w/o ignore border, and stride <= pool_size
    test_same((1, 1, 5, 6), (2, 3), (1, 1), (0, 0), False)
    test_same((1, 1, 5, 6), (2, 3), (2, 2), (0, 0), False)
    test_same((1, 1, 1, 1), (2, 3), (2, 2), (0, 0), False)

    # tests w/o ignore border, and stride > pool_size
    test_same((1, 1, 5, 6), (2, 3), (3, 3), (0, 0), False)
    test_same((1, 1, 5, 6), (2, 3), (3, 3), (0, 0), False)
    test_same((1, 1, 1, 1), (2, 3), (3, 3), (0, 0), False)


def test_pool_output_shape_3d():
    def test_same(input_shape, local_sizes, strides, pads, ignore_border, ans):
        res = tn.downsample.pool_output_shape(
            input_shape,
            (2, 3, 4),
            local_sizes,
            strides,
            pads,
            ignore_border,
        )
        print(ans, res)
        np.testing.assert_equal(ans, res)

    test_same((1, 1, 2, 2, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), False,
              ans=(1, 1, 1, 1, 1))


def test_pool_output_shape_custom_pool_2d_node():
    def test_same(input_shape, local_sizes, strides, pads, ignore_border):
        res = tn.downsample.pool_output_shape(
            input_shape,
            (2, 3),
            local_sizes,
            strides,
            pads,
            ignore_border,
        )
        # pool2d node assumes 0 padding
        assert pads == (0, 0)
        # pool2d node assumes ignoring border
        assert ignore_border
        network = tn.SequentialNode(
            "s",
            [tn.ConstantNode("c",
                             value=np.random.randn(*input_shape).astype(fX)),
             tn.CustomPool2DNode("p",
                                 pool_function=T.mean,
                                 pool_size=local_sizes,
                                 stride=strides,
                                 )]
        ).network()
        ans = network["p"].get_vw("default").variable.shape.eval()
        print(ans, res)
        np.testing.assert_equal(ans, res)

    test_same((3, 4, 5, 6), (2, 3), (1, 1), (0, 0), True)
    test_same((3, 4, 5, 6), (2, 3), (2, 2), (0, 0), True)
    test_same((3, 4, 1, 1), (2, 3), (2, 2), (0, 0), True)


def test_feature_pool_node_serialization():
    tn.check_serialization(tn.FeaturePoolNode("a"))


def test_maxout_node_serialization():
    tn.check_serialization(tn.MaxoutNode("a"))


def test_custom_pool_2d_node_serialization():
    tn.check_serialization(tn.CustomPool2DNode("a"))


def test_mean_pool_2d_node_serialization():
    tn.check_serialization(tn.MeanPool2DNode("a"))


def test_global_pool_node_serialization():
    tn.check_serialization(tn.CustomGlobalPoolNode("a"))


def test_maxout_hyperparameters():
    nt.assert_equal(
        set(tn.FeaturePoolNode.hyperparameter_names),
        set(tn.MaxoutNode.hyperparameter_names + ("pool_function",)))


def test_maxout_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 15)),
         tn.MaxoutNode("m", num_pieces=5)]).network()

    fn = network.function(["i"], ["m"])
    x = np.arange(15).astype(fX).reshape(1, 15)
    np.testing.assert_equal(fn(x)[0],
                            np.array([[4, 9, 14]], dtype=fX))


def test_mean_pool_2d_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 4, 4)),
         tn.MeanPool2DNode("m", pool_size=(2, 2))]).network()
    fn = network.function(["i"], ["m"])
    x = np.arange(16).astype(fX).reshape(1, 1, 4, 4)
    ans = np.array([[[[0 + 1 + 4 + 5, 2 + 3 + 6 + 7],
                      [8 + 9 + 12 + 13, 10 + 11 + 14 + 15]]]], dtype=fX) / 4
    np.testing.assert_equal(ans, fn(x)[0])
    nt.assert_equal(ans.shape,
                    network["m"].get_vw("default").shape)


def test_max_pool_2d_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 4, 4)),
         tn.MaxPool2DNode("m", pool_size=(2, 2))]).network()
    fn = network.function(["i"], ["m"])
    x = np.arange(16).astype(fX).reshape(1, 1, 4, 4)
    ans = np.array([[[[5, 7],
                      [13, 15]]]], dtype=fX)
    np.testing.assert_equal(ans, fn(x)[0])
    nt.assert_equal(ans.shape,
                    network["m"].get_vw("default").shape)


# sum pool doesn't work with cudnn
if "gpu" not in theano.config.device:
    def test_sum_pool_2d_node():
        network = tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=(1, 1, 4, 4)),
             tn.SumPool2DNode("m", pool_size=(2, 2))]).network()
        fn = network.function(["i"], ["m"])
        x = np.arange(16).astype(fX).reshape(1, 1, 4, 4)
        ans = np.array([[[[0 + 1 + 4 + 5, 2 + 3 + 6 + 7],
                          [8 + 9 + 12 + 13, 10 + 11 + 14 + 15]]]], dtype=fX)
        np.testing.assert_equal(ans, fn(x)[0])
        nt.assert_equal(ans.shape,
                        network["m"].get_vw("default").shape)


def test_custom_global_pool_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(6, 5, 4, 3)),
         tn.CustomGlobalPoolNode("gp", pool_function=T.mean)]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.random.randn(6, 5, 4, 3).astype(fX)
    ans = x.mean(axis=(2, 3))
    np.testing.assert_allclose(ans,
                               fn(x)[0],
                               rtol=1e-5,
                               atol=1e-7)


def test_global_mean_pool_2d_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(6, 5, 4, 3)),
         tn.GlobalMeanPool2DNode("gp")]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.random.randn(6, 5, 4, 3).astype(fX)
    ans = x.mean(axis=(2, 3))
    np.testing.assert_allclose(ans,
                               fn(x)[0],
                               rtol=1e-5,
                               atol=1e-7)
