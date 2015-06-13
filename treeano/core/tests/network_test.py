import nose.tools as nt
from treeano import core
import treeano.nodes as tn


def test_find_hyperparameters():
    class FooNode(core.WrapperNodeImpl):
        hyperparameter_names = ("a", "b", "c")

    last_foo = FooNode("last", [tn.InputNode("i", shape=(1,))], b=1)
    mid_foo = FooNode("mid", [last_foo], a=2, c=3)
    top_foo = FooNode("top", [mid_foo], a=4, b=5, c=6)

    network = top_foo.network(
        default_hyperparameters={"a": 7, "b": 8, "c": 9},
        override_hyperparameters={"a": 10, "b": 11, "c": 12}
    )

    nt.assert_equal([10, 11, 12, 1, 2, 3, 4, 5, 6, 13, 7, 8, 9],
                    list(network["last"].find_hyperparameters(["a", "b", "c"],
                                                              13)))
    nt.assert_equal([10, 11, 12, 2, 3, 4, 5, 6, 13, 7, 8, 9],
                    list(network["mid"].find_hyperparameters(["a", "b", "c"],
                                                             13)))
    nt.assert_equal([10, 11, 12, 4, 5, 6, 13, 7, 8, 9],
                    list(network["top"].find_hyperparameters(["a", "b", "c"],
                                                             13)))
