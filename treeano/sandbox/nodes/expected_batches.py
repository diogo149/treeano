import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


class BaseExpectedBatchesNode(treeano.NodeImpl):

    hyperparameter_names = ("expected_batches",)

    def init_state(self, network):
        expected_batches = network.find_hyperparameter(["expected_batches"])
        expected_batches = treeano.utils.as_fX(expected_batches)
        batch_idx = network.create_vw(
            name="batch_idx",
            is_shared=True,
            shape=(),
            tags={"state"},
            default_inits=[],
        ).variable

        network.create_vw(
            name="progress",
            variable=batch_idx / expected_batches,
            shape=(),
            tags={},
        )

    def new_update_deltas(self, network):
        batch_idx = network.get_vw("batch_idx").variable
        ud = treeano.UpdateDeltas()
        ud[batch_idx] = treeano.utils.as_fX(1)
        return ud

    def get_progress(self, network):
        return network.get_vw("progress").variable


@treeano.register_node("linear_interpolation")
class LinearInterpolationNode(BaseExpectedBatchesNode):

    """
    interpolates outputs between 2 nodes
    """

    hyperparameter_names = ("expected_batches",
                            "start_percent",
                            "end_percent")
    children_container = treeano.core.DictChildrenContainerSchema(
        early=treeano.core.ChildContainer,
        late=treeano.core.ChildContainer,
    )
    input_keys = ("early", "late")

    def init_state(self, network):
        super(LinearInterpolationNode, self).init_state(network)

        children = self.raw_children()
        early = children["early"]
        late = children["late"]

        network.forward_input_to(early.name)
        network.forward_input_to(late.name)

        network.take_output_from(early.name, to_key="early")
        network.take_output_from(late.name, to_key="late")

    def compute_output(self, network, early_vw, late_vw):
        progress = self.get_progress(network)
        start_percent = network.find_hyperparameter(["start_percent"], 0)
        end_percent = network.find_hyperparameter(["end_percent"], 0.2)
        late_gate = T.clip((progress - start_percent) / (end_percent - start_percent),
                           0,
                           1)
        out_var = (early_vw.variable * (1 - late_gate)
                   + late_vw.variable * late_gate)

        if False:
            # optimization for not computing early branch
            out_var = theano.ifelse.ifelse(late_gate >= 1,
                                           late_vw.variable,
                                           out_var)

        # create vw for monitoring
        network.create_vw(
            name="late_gate",
            variable=late_gate,
            shape=(),
            tags={"hyperparameter", "monitor"},
        )

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


@treeano.register_node("gradual_relu")
class GradualReLUNode(BaseExpectedBatchesNode):

    hyperparameter_names = ("expected_batches",
                            "start_percent",
                            "end_percent")

    def compute_output(self, network, in_vw):
        progress = self.get_progress(network)
        start_percent = network.find_hyperparameter(["start_percent"], 0)
        end_percent = network.find_hyperparameter(["end_percent"], 0.2)
        late_gate = T.clip((progress - start_percent) / (end_percent - start_percent),
                           0,
                           1)
        alpha = 1 - late_gate
        out_var = treeano.utils.rectify(in_vw.variable,
                                        negative_coefficient=alpha)

        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"},
        )

        # create vw for monitoring
        network.create_vw(
            name="alpha",
            variable=alpha,
            shape=(),
            tags={"hyperparameter", "monitor"},
        )


@treeano.register_node("gradual_dropout")
class GradualDropoutNode(BaseExpectedBatchesNode):

    hyperparameter_names = ("expected_batches",
                            "start_percent",
                            "end_percent",
                            "dropout_probability",
                            "p")
    children_container = treeano.core.ChildContainer
    input_keys = ("child_output",)

    def init_state(self, network):
        super(GradualDropoutNode, self).init_state(network)

        # forward to child
        child, = self.architecture_children()
        network.forward_input_to(child.name)
        network.take_output_from(child.name, to_key="child_output")

        # calculate probability
        progress = self.get_progress(network)
        start_percent = network.find_hyperparameter(["start_percent"], 0)
        end_percent = network.find_hyperparameter(["end_percent"], 0.2)
        p = network.find_hyperparameter(["dropout_probability", "p"])

        late_gate = T.clip((progress - start_percent) / (end_percent - start_percent),
                           0,
                           1)

        prob = p * late_gate

        network.set_hyperparameter(child.name, "dropout_probability", prob)

        # create vw for monitoring
        network.create_vw(
            name="dropout_probability",
            variable=prob,
            shape=(),
            tags={"hyperparameter", "monitor"},
        )


def GradualBatchNormalization(name, **kwargs):
    from treeano.sandbox.nodes import batch_normalization as bn
    return tn.HyperparameterNode(
        name,
        LinearInterpolationNode(
            name + "_interpolate",
            {"early": bn.BatchNormalizationNode(name + "_bn"),
             "late": tn.IdentityNode(name + "_identity")}),
        **kwargs)


@treeano.register_node("scale_hyperparameter")
class ScaleHyperparameterNode(BaseExpectedBatchesNode):

    hyperparameter_names = ("hyperparameter",
                            "start_percent",
                            "end_percent",
                            "start_scale",
                            "end_scale")
    children_container = treeano.core.ChildContainer
    input_keys = ("child_output",)

    def init_state(self, network):
        super(ScaleHyperparameterNode, self).init_state(network)

        # forward to child
        child, = self.architecture_children()
        network.forward_input_to(child.name)
        network.take_output_from(child.name, to_key="child_output")

        # calculate probability
        progress = self.get_progress(network)
        start_percent = network.find_hyperparameter(["start_percent"], 0)
        end_percent = network.find_hyperparameter(["end_percent"], 0.2)

        late_gate = T.clip((progress - start_percent) / (end_percent - start_percent),
                           0,
                           1)

        hyperparameter = network.find_hyperparameter(["hyperparameter"])
        prev_hp = network.find_hyperparameter([hyperparameter])
        start_scale = network.find_hyperparameter(["start_scale"])
        end_scale = network.find_hyperparameter(["end_scale"])

        scale = end_scale * late_gate + start_scale * (1 - late_gate)
        value = prev_hp * scale

        network.set_hyperparameter(self.name, hyperparameter, value)

        # create vw for monitoring
        network.create_vw(
            name="scale",
            variable=scale,
            shape=(),
            tags={"hyperparameter", "monitor"},
        )
        network.create_vw(
            name="hyperparameter",
            variable=value,
            shape=(),
            tags={"hyperparameter", "monitor"},
        )


@treeano.register_node("linear_hyperparameter")
class LinearHyperparameterNode(BaseExpectedBatchesNode):

    hyperparameter_names = ("hyperparameter",
                            "start_percent",
                            "end_percent",
                            "start_value",
                            "end_value")
    children_container = treeano.core.ChildContainer
    input_keys = ("child_output",)

    def init_state(self, network):
        super(LinearHyperparameterNode, self).init_state(network)

        # forward to child
        child, = self.architecture_children()
        network.forward_input_to(child.name)
        network.take_output_from(child.name, to_key="child_output")

        # calculate probability
        progress = self.get_progress(network)
        start_percent = network.find_hyperparameter(["start_percent"], 0)
        end_percent = network.find_hyperparameter(["end_percent"], 0.2)

        late_gate = T.clip((progress - start_percent) / (end_percent - start_percent),
                           0,
                           1)

        hyperparameter = network.find_hyperparameter(["hyperparameter"])
        start_value = network.find_hyperparameter(["start_value"])
        end_value = network.find_hyperparameter(["end_value"])

        value = end_value * late_gate + start_value * (1 - late_gate)

        network.set_hyperparameter(self.name, hyperparameter, value)

        # create vw for monitoring
        network.create_vw(
            name="hyperparameter",
            variable=value,
            shape=(),
            tags={"hyperparameter", "monitor"},
        )
