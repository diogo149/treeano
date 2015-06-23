import os
import shutil
import tempfile

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

import canopy

fX = theano.config.floatX


def test_pickle_unpickle_network():
    temp_dir = tempfile.mkdtemp()
    dirname = os.path.join(temp_dir, "network")
    try:
        n1 = tn.SequentialNode(
            "seq",
            [tn.InputNode("i", shape=(10, 100)),
             tn.LinearMappingNode(
                 "lm",
                output_dim=15,
                inits=[treeano.inits.NormalWeightInit()])]
        ).network()

        fn1 = n1.function(["i"], ["lm"])
        x = np.random.randn(10, 100).astype(fX)
        canopy.serialization.pickle_network(n1, dirname)
        n2 = canopy.serialization.unpickle_network(dirname)
        fn2 = n2.function(["i"], ["lm"])
        np.testing.assert_equal(fn1(x), fn2(x))
    finally:
        shutil.rmtree(temp_dir)
