import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import expected_batches as eb

fX = theano.config.floatX


def test_scale_hyperparameter():
    network = tn.HyperparameterNode(
        "hp",
        eb.ScaleHyperparameterNode(
            "scale",
            tn.ConstantNode("c")),
        value=42.0,
        hyperparameter="value",
        start_percent=0.,
        end_percent=1.0,
        scale=0.1,
        expected_batches=2,
    )

    fn = network.function([], ["c"], include_updates=True)

    nt.assert_equal(42.0,
                    fn()[0])
    nt.assert_equal(42.0 * 0.55,
                    fn()[0])
    nt.assert_equal(42.0 * 0.1,
                    fn()[0])
    nt.assert_equal(42.0 * 0.1,
                    fn()[0])
