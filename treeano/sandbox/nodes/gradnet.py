import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("grad_net_interpolation")
class GradNetInterpolationNode(treeano.NodeImpl):

    """
    interpolates outputs between 2 nodes
    """

    hyperparameter_names = ("late_gate",)
    children_container = treeano.core.DictChildrenContainerSchema(
        early=treeano.core.ChildContainer,
        late=treeano.core.ChildContainer,
    )
    input_keys = ("early", "late")

    def init_state(self, network):
        children = self.raw_children()
        early = children["early"]
        late = children["late"]

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
            tags={"output"},
        )


@treeano.register_node("grad_net_optimizer_interpolation")
class _GradNetOptimizerInterpolationNode(treeano.Wrapper1NodeImpl):

    hyperparameter_names = ("late_gate",
                            "gradnet_epsilon",
                            "epsilon",
                            "multiplicative_inverse_for_early_gate")

    def init_state(self, network):
        super(_GradNetOptimizerInterpolationNode, self).init_state(network)
        epsilon = network.find_hyperparameter(["gradnet_epsilon",
                                               "epsilon"],
                                              1e-3)
        late_gate = network.find_hyperparameter(["late_gate"], 1)
        late_gate = treeano.utils.as_fX(late_gate)
        # NOTE: late gate cannot be 0 because the early gate is divide by it
        # AND multiplied by it. Clipping only for the early gate will cause
        # no updates to occur.
        late_gate = T.clip(late_gate, epsilon, 1)

        use_multiplicative_inverse = network.find_hyperparameter(
            ["multiplicative_inverse_for_early_gate"], False)
        if use_multiplicative_inverse:
            early_gate = epsilon / late_gate
        else:
            early_gate = 1 - late_gate

        network.set_hyperparameter(self.name + "_late_update_scale",
                                   "update_scale_factor",
                                   late_gate)
        network.set_hyperparameter(self.name + "_early_update_scale",
                                   "update_scale_factor",
                                   # these updates are also multiplied by
                                   # late_gate later on, so rescale them
                                   early_gate / late_gate)


def GradNetOptimizerInterpolationNode(name,
                                      children,
                                      early,
                                      late,
                                      **kwargs):
    """
    interpolates updates from 2 optimizers nodes

    NOTE: this is a hack to take in node constructors as arguments
    """
    assert set(children.keys()) == {"subtree", "cost"}
    subtree = children["subtree"]
    cost = children["cost"]

    cost_ref = tn.ReferenceNode(name + "_costref", reference=cost.name)
    late_subtree = tn.UpdateScaleNode(name + "_late_update_scale", subtree)
    late_node = late(name + "_late", {"subtree": late_subtree, "cost": cost})
    early_subtree = tn.UpdateScaleNode(name + "_early_update_scale", late_node)
    early_node = early(name + "_early",
                       {"subtree": early_subtree, "cost": cost_ref})
    # NOTE: need separate node to forward hyperparameter
    return _GradNetOptimizerInterpolationNode(name, early_node, **kwargs)


def GradualSimpleBatchNormalizationNode(name):
    from treeano.sandbox.nodes import batch_normalization as bn
    return GradNetInterpolationNode(
        name,
        {"early": bn.SimpleBatchNormalizationNode(name + "_bn"),
         "late": tn.IdentityNode(name + "_identity")})

GradualBNNode = GradualSimpleBatchNormalizationNode
