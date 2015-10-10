import numpy as np
import theano
import theano.tensor as T

import treeano


fX = theano.config.floatX


def test_constant_init():
    class DummyNode(treeano.NodeImpl):

        input_keys = ()

        def init_state(self, network):
            network.set_hyperparameter(self.name,
                                       "inits",
                                       [treeano.inits.ConstantInit(1)])

        def compute_output(self, network):
            inits = network.find_hyperparameter(["inits"])
            network.create_vw(
                "default",
                is_shared=True,
                shape=(1, 2, 3),
                inits=inits,
            )

    network = DummyNode("dummy").network()
    fn = network.function([], ["dummy"])
    np.testing.assert_allclose(fn()[0],
                               np.ones((1, 2, 3)).astype(fX),
                               rtol=1e-5,
                               atol=1e-8)
