import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import batch_fold

fX = theano.config.floatX


@treeano.register_node("grad_net_interpolation")
class GradNetInterpolationNode(treeano.NodeImpl):

    """
    interpolates outputs between 2 nodes
    """

    hyperparameter_names = ("late_gate",)

    children_container = treeano.core.DictChildrenContainerSchema(
        early=treeano.core.ChildContainer,
        late=treeano.core.ChildContainer)

    input_keys = ("early", "late")

    def init_state(self, network):

        early = self._children["early"].children
        late = self._children["late"].children

        network.forward_input_to(early.name)
        network.forward_input_to(late.name)

        network.take_output_from(early.name, to_key="early")
        network.take_output_from(late.name, to_key="late")

    def compute_output(self, network, early_vw, late_vw):
        late_gate = network.find_hyperparameter(["late_gate"], 1)
        out_var = (early_vw.variable * (1 - late_gate)
                   + late_vw.variable * late_gate)

        out_shape = []
        assert early_vw.ndim == late_vw.ndim

        for e, l in zip(early_vw.shape, late_vw.shape):
            if e is None and l is None:
                out_shape.append(None)
            elif e is None:
                out_shape.append(l)
            elif l is None:
                out_shape.append(e)
            else:
                assert e == l
                out_shape.append(e)

        network.create_vw(
            "default",
            variable=out_var,
            shape=tuple(out_shape),
            tags={"output"})
