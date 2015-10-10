"""
some not-actually-useful nodes (if they are, move them elsewhere) mostly
for tests and code samples
"""

from .. import core


@core.register_node("constant_updater")
class ConstantUpdaterNode(core.Wrapper1NodeImpl):

    """
    provides updates as a constant value
    """

    hyperparameter_names = ("value",)

    def new_update_deltas(self, network):
        value = network.find_hyperparameter(["value"])
        parameters = network.find_vws_in_subtree(tags=["parameter"])
        return core.UpdateDeltas({p.variable: value for p in parameters})


@core.register_node("scalar_sum")
class ScalarSumNode(core.NodeImpl):

    """
    sums up its input into a scalar
    """

    def compute_output(self, network, in_vw):
        network.create_vw(
            "default",
            variable=in_vw.variable.sum(),
            shape=(),
            tags={"output"},
        )
