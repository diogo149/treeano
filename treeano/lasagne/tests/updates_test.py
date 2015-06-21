import treeano.nodes as tn
import treeano.lasagne.nodes as tl


def test_sgd_node():
    tn.test_utils.check_updates_node(tl.SGDNode, learning_rate=0.01)


def test_nesterov_momentum_node():
    tn.test_utils.check_updates_node(tl.NesterovMomentumNode,
                                     learning_rate=0.01)
