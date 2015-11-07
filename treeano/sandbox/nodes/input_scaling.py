import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import treeano.sandbox.utils


@treeano.register_node("clip_scaling")
class ClipScalingNode(treeano.NodeImpl):

    hyperparameter_names = ("learnable",
                            "mins",
                            "maxs")

    def compute_output(self, network, in_vw):
        learnable = network.find_hyperparameter(["learnable"], False)
        mins = network.find_hyperparameter(["mins"])
        maxs = network.find_hyperparameter(["maxs"])
        assert mins.ndim == maxs.ndim == 1
        assert mins.shape == maxs.shape
        mins = treeano.utils.as_fX(mins)
        maxs = treeano.utils.as_fX(maxs)
        num_scales = mins.shape[0]

        if learnable:
            mins_var = network.create_vw(
                "mins",
                shape=mins.shape,
                is_shared=True,
                tags={"parameter"},
                default_inits=[treeano.inits.ConstantInit(mins)],
            ).variable
            maxs_var = network.create_vw(
                "maxs",
                shape=maxs.shape,
                is_shared=True,
                tags={"parameter"},
                default_inits=[treeano.inits.ConstantInit(maxs)],
            ).variable
        else:
            if treeano.utils.is_variable(mins):
                mins_var = mins
            else:
                mins_var = T.constant(mins)

            if treeano.utils.is_variable(maxs):
                maxs_var = maxs
            else:
                maxs_var = T.constant(maxs)

        in_pattern = list(range(in_vw.ndim))
        # insert after channel dim
        in_pattern.insert(2, "x")

        param_pattern = ["x"] * in_vw.ndim
        param_pattern.insert(2, 0)

        in_b = in_vw.variable.dimshuffle(*in_pattern)
        mins_b = mins_var.dimshuffle(*param_pattern)
        maxs_b = maxs_var.dimshuffle(*param_pattern)

        range_b = maxs_b - mins_b
        # TODO constrain range to be > 0?
        clipped = T.clip(in_b - mins_b, 0, range_b)
        scaled = clipped / range_b

        # reshape newly created dim into dim 1
        out_ss = list(in_vw.symbolic_shape())
        out_ss[1] *= num_scales
        out_var = scaled.reshape(tuple(out_ss))

        out_shape = list(in_vw.shape)
        if out_shape[1] is not None:
            out_shape[1] *= num_scales
        out_shape = tuple(out_shape)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={}
        )


@treeano.register_node("tanh_scaling")
class TanhScalingNode(treeano.NodeImpl):

    hyperparameter_names = ("learnable",
                            "means",
                            "scales")

    def compute_output(self, network, in_vw):
        learnable = network.find_hyperparameter(["learnable"], False)
        means = network.find_hyperparameter(["means"])
        scales = network.find_hyperparameter(["scales"])
        assert means.ndim == scales.ndim == 1
        assert means.shape == scales.shape
        means = treeano.utils.as_fX(means)
        scales = treeano.utils.as_fX(scales)
        num_scales = means.shape[0]

        if learnable:
            means_var = network.create_vw(
                "means",
                shape=means.shape,
                is_shared=True,
                tags={"parameter"},
                default_inits=[treeano.inits.ConstantInit(means)],
            ).variable
            scales_var = network.create_vw(
                "scales",
                shape=scales.shape,
                is_shared=True,
                tags={"parameter"},
                default_inits=[treeano.inits.ConstantInit(scales)],
            ).variable
        else:
            if treeano.utils.is_variable(means):
                means_var = means
            else:
                means_var = T.constant(means)

            if treeano.utils.is_variable(scales):
                scales_var = scales
            else:
                scales_var = T.constant(scales)

        in_pattern = list(range(in_vw.ndim))
        # insert after channel dim
        in_pattern.insert(2, "x")

        param_pattern = ["x"] * in_vw.ndim
        param_pattern.insert(2, 0)

        in_b = in_vw.variable.dimshuffle(*in_pattern)
        means_b = means_var.dimshuffle(*param_pattern)
        scales_b = scales_var.dimshuffle(*param_pattern)

        # TODO constrain scales to be > 0?
        scaled = T.tanh((in_b - means_b) / scales_b)

        # reshape newly created dim into dim 1
        out_ss = list(in_vw.symbolic_shape())
        out_ss[1] *= num_scales
        out_var = scaled.reshape(tuple(out_ss))

        out_shape = list(in_vw.shape)
        if out_shape[1] is not None:
            out_shape[1] *= num_scales
        out_shape = tuple(out_shape)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@treeano.register_node("rbf_scaling")
class RBFScalingNode(treeano.NodeImpl):

    hyperparameter_names = ("learnable",
                            "means",
                            "scales")

    def compute_output(self, network, in_vw):
        learnable = network.find_hyperparameter(["learnable"], False)
        means = network.find_hyperparameter(["means"])
        scales = network.find_hyperparameter(["scales"])
        assert means.ndim == scales.ndim == 1
        assert means.shape == scales.shape
        means = treeano.utils.as_fX(means)
        scales = treeano.utils.as_fX(scales)
        num_scales = means.shape[0]

        if learnable:
            means_var = network.create_vw(
                "means",
                shape=means.shape,
                is_shared=True,
                tags={"parameter"},
                default_inits=[treeano.inits.ConstantInit(means)],
            ).variable
            scales_var = network.create_vw(
                "scales",
                shape=scales.shape,
                is_shared=True,
                tags={"parameter"},
                default_inits=[treeano.inits.ConstantInit(scales)],
            ).variable
        else:
            if treeano.utils.is_variable(means):
                means_var = means
            else:
                means_var = T.constant(means)

            if treeano.utils.is_variable(scales):
                scales_var = scales
            else:
                scales_var = T.constant(scales)

        in_pattern = list(range(in_vw.ndim))
        # insert after channel dim
        in_pattern.insert(2, "x")

        param_pattern = ["x"] * in_vw.ndim
        param_pattern.insert(2, 0)

        in_b = in_vw.variable.dimshuffle(*in_pattern)
        means_b = means_var.dimshuffle(*param_pattern)
        scales_b = scales_var.dimshuffle(*param_pattern)

        # TODO constrain scales to be > 0?
        scaled = T.exp(-T.sqr(in_b - means_b) / scales_b)

        # reshape newly created dim into dim 1
        out_ss = list(in_vw.symbolic_shape())
        out_ss[1] *= num_scales
        out_var = scaled.reshape(tuple(out_ss))

        out_shape = list(in_vw.shape)
        if out_shape[1] is not None:
            out_shape[1] *= num_scales
        out_shape = tuple(out_shape)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )
