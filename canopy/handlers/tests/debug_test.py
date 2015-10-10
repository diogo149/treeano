import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_output_nanguard():
    def output_nanguard_fn(a):
        class CustomNode(treeano.NodeImpl):

            def compute_output(self, network, in_vw):
                network.create_vw(
                    "default",
                    variable=in_vw.variable / a,
                    shape=in_vw.shape
                )

        network = tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=()),
             CustomNode("c")]
        ).network()

        return canopy.handlers.handled_fn(
            network,
            [canopy.handlers.output_nanguard()],
            {"x": "i"},
            {"out": "s"})

    fn1 = output_nanguard_fn(3)
    np.testing.assert_equal(fn1({"x": 3}), {"out": np.array(1)})
    np.testing.assert_equal(fn1({"x": -6}), {"out": np.array(-2)})

    fn2 = output_nanguard_fn(0)
    try:
        fn2({"x": 3})
    except Exception as e:
        assert e.args[0]["error_type"] == "inf"
        np.testing.assert_equal(e.args[0]["value"], np.array(np.inf))
    else:
        assert False

    try:
        fn2({"x": -6})
    except Exception as e:
        nt.assert_equal(e.args[0]["error_type"], "inf")
        np.testing.assert_equal(e.args[0]["value"], np.array(-np.inf))
    else:
        assert False

    try:
        fn2({"x": 0})
    except Exception as e:
        nt.assert_equal(e.args[0]["error_type"], "nan")
        np.testing.assert_equal(e.args[0]["value"], np.array(np.nan))
    else:
        assert False

    try:
        fn1({"x": 6e10})
    except Exception as e:
        nt.assert_equal(e.args[0]["error_type"], "big")
        np.testing.assert_allclose(e.args[0]["value"],
                                   np.array(2e10),
                                   rtol=1e-5)
    else:
        assert False


def test_nanguardmode():
    def nanguardmode_fn(a):
        class CustomNode(treeano.NodeImpl):

            def compute_output(self, network, in_vw):
                network.create_vw(
                    "default",
                    variable=in_vw.variable / a,
                    shape=in_vw.shape
                )

        network = tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=()),
             CustomNode("c")]
        ).network()

        return canopy.handlers.handled_fn(
            network,
            [canopy.handlers.nanguardmode()],
            {"x": "i"},
            {"out": "s"})

    fn1 = nanguardmode_fn(3)
    np.testing.assert_equal(fn1({"x": 3}), {"out": np.array(1)})
    np.testing.assert_equal(fn1({"x": -6}), {"out": np.array(-2)})

    @nt.raises(AssertionError)
    def raises_fn1(x):
        fn1({"x": x})

    raises_fn1(6e10)

    fn2 = nanguardmode_fn(0)

    @nt.raises(AssertionError)
    def raises_fn2(x):
        fn2({"x": x})

    raises_fn2(3)
    raises_fn2(-6)
    raises_fn2(0)
    raises_fn2(6e10)


def test_save_last_inputs_and_networks():

    class StateDiffNode(treeano.NodeImpl):

        def compute_output(self, network, in_vw):
            foo_vw = network.create_vw(
                "foo",
                shape=(),
                is_shared=True,
                tags={"parameter", "weight"},
                inits=[]
            )
            network.create_vw(
                "default",
                variable=abs(in_vw.variable - foo_vw.variable),
                shape=()
            )

    network = tn.AdamNode(
        "adam",
        {"subtree": tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=()),
             StateDiffNode("ss")]),
         "cost": tn.ReferenceNode("r", reference="s")}
    ).network()
    # eagerly create shared variables
    network.build()

    save_handler = canopy.handlers.save_last_inputs_and_networks(5)
    fn = canopy.handlers.handled_fn(
        network,
        [save_handler],
        {"x": "i"},
        {"out": "s"},
        include_updates=True)

    inputs = [{"x": treeano.utils.as_fX(np.random.randn())} for _ in range(10)]
    outputs = [fn(i) for i in inputs]

    nt.assert_equal(save_handler.inputs_, inputs[-5:])

    # PY3: calling list on zip to make it eager
    # otherwise, save_handler.value_dicts_ looks at the mutating
    # value ducts
    for value_dict, i, o in list(zip(save_handler.value_dicts_,
                                     inputs[-5:],
                                     outputs[-5:])):
        canopy.network_utils.load_value_dict(network, value_dict)
        nt.assert_equal(o, fn(i))


def test_network_nanguard():
    class CustomNode(treeano.NodeImpl):
        input_keys = ()

        def compute_output(self, network):
            network.create_vw(
                "default",
                is_shared=True,
                shape=(),
                inits=[]
            )

    network = CustomNode("c").network()
    # build eagerly to share weights
    network.build()

    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.network_nanguard()],
        {},
        {})

    vw = network["c"].get_variable("default")
    for x in [3, 4, 1e9, 9e9, -9e9, 0]:
        vw.variable.set_value(treeano.utils.as_fX(x))
        fn({})

    for x in [np.inf, -np.inf, np.nan, 2e10]:
        vw.variable.set_value(treeano.utils.as_fX(x))
        nt.raises(Exception)(lambda x: fn(x))({})
