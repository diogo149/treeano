import numpy as np
import theano
import theano.tensor as T

import treeano


fX = theano.config.floatX


def test_constant_init():
    class DummyNode(treeano.NodeImpl):

        input_keys = ()

        def get_hyperparameter(self, network, hyperparameter_name):
            if hyperparameter_name == "shared_initializations":
                return [treeano.inits.ConstantInit(1)]
            else:
                return super(DummyNode, self).get_hyperparameter(
                    network,
                    hyperparameter_name)

        def compute_output(self, network):
            network.create_variable(
                "default",
                is_shared=True,
                shape=(1, 2, 3),
            )

    network = DummyNode("dummy").build()
    fn = network.function([], ["dummy"])
    np.testing.assert_allclose(fn()[0],
                               np.ones((1, 2, 3)).astype(fX),
                               rtol=1e-5,
                               atol=1e-8)
