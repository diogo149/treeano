"""
from
"Spatial Transformer Networks"
http://arxiv.org/abs/1506.02025

differentiable attention mechanism
based on:
- https://github.com/skaae/transformer_network
- http://pogo:12348/notebooks/spatial_transformer_network.ipynb
"""

import numpy as np
import skimage.transform
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def target_grid(output_shape):
    """
    returns grid of relative homogeneous coordinates that fit in the given
    output_shape
    - relative homogeneous coordinates meaning scaled from -1 to 1
    - result shape = (3, num_points)
    - num_points = prod(output_shape)
    """
    width, height = output_shape
    x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                           np.linspace(-1, 1, height))
    ones = np.ones(width * height)
    grid = np.vstack([x_t.flatten(),
                      y_t.flatten(),
                      ones])
    assert grid.shape == (3, np.prod(output_shape))
    return grid.astype(fX)


def affine_matrix_batch(theta):
    # create identity with shape (1, 2, 3)
    # - the 1 is for broadcasting that dimension to batch_size
    affine_identity = np.array([[[1, 0, 0],
                                 [0, 1, 0]]],
                               dtype=fX)
    # reshape theta to (batch_size, 2, 3)
    theta_reshaped = theta.reshape((-1, 2, 3))
    # add in identity matrix, since that is a good default
    return affine_identity + theta_reshaped


def affine_warp_coordinates(transform_matrix_batch, grid):
    # apply affine matrix
    new_points = transform_matrix_batch.dot(grid)
    x = new_points[:, 0]
    y = new_points[:, 1]
    return x, y


def warp_bilinear_interpolation(orig_img, x, y, out_height, out_width):
    # shuffle channel dim to last dimension, since we want to apply the same
    # transform to the whole dim
    img = orig_img.dimshuffle(0, 2, 3, 1)
    # flatten batch dims
    x = x.flatten()
    y = y.flatten()
    # *_f are floats
    num_batch, height, width, num_channels = img.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # scale indices from [-1, 1] to [0, width/height].
    x = (x + 1) / 2 * width_f
    y = (y + 1) / 2 * height_f

    # Clip indices to ensure they are not out of bounds.
    max_x = width_f - 1
    max_y = height_f - 1
    # TODO add monitoring to out of bounds points
    x0 = T.clip(x, 0, max_x)
    x1 = T.clip(x + 1, 0, max_x)
    y0 = T.clip(y, 0, max_y)
    y1 = T.clip(y + 1, 0, max_y)

    # We need floatX for interpolation and int64 for indexing.
    x0_f = T.floor(x0)
    x1_f = T.floor(x1)
    y0_f = T.floor(y0)
    y1_f = T.floor(y1)
    x0 = T.cast(x0, 'int64')
    x1 = T.cast(x1, 'int64')
    y0 = T.cast(y0, 'int64')
    y1 = T.cast(y1, 'int64')

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width * height
    base = T.repeat(T.arange(num_batch, dtype='int64') * dim1,
                    out_height * out_width)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    img_flat = img.reshape((-1, num_channels))
    Ia = img_flat[idx_a]
    Ib = img_flat[idx_b]
    Ic = img_flat[idx_c]
    Id = img_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f - x) * (y1_f - y)).dimshuffle(0, 'x')
    wb = ((x1_f - x) * (y - y0_f)).dimshuffle(0, 'x')
    wc = ((x - x0_f) * (y1_f - y)).dimshuffle(0, 'x')
    wd = ((x - x0_f) * (y - y0_f)).dimshuffle(0, 'x')
    output_2d = T.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
    output_4d = T.reshape(output_2d,
                          (num_batch, out_height, out_width, num_channels))
    # convert back from b01c (batch, dim0, dim1, channels)
    # to bc01 (batch, channels, dim0, dim1)
    output = output_4d.dimshuffle(0, 3, 1, 2)
    return output


@treeano.register_node("affine_spatial_transformer")
class AffineSpatialTransformerNode(treeano.Wrapper1NodeImpl):

    input_keys = ("default", "final_child_output")
    hyperparameter_names = ("output_shape",)

    def compute_output(self, network, in_vw, theta_vw):
        output_shape = network.find_hyperparameter(["output_shape"])
        assert len(output_shape) == 2

        theta = theta_vw.variable
        # calculate grid in homogeneous coordinates (x[t],y[t],1)
        grid = target_grid(output_shape)
        # calculate affine transform matrix for each element in the batch
        affine_parameters = affine_matrix_batch(theta)
        # map target coords to source coords: x[t],y[t] -> x[s],y[s]
        x_s, y_s = affine_warp_coordinates(affine_parameters, grid)
        # get new image
        out_var = warp_bilinear_interpolation(in_vw.variable,
                                              x_s,
                                              y_s,
                                              *output_shape)

        network.create_vw(
            "affine_parameters",
            variable=affine_parameters,
            shape=(in_vw.shape[0], 2, 3),
            tags={},
        )
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape[:2] + output_shape,
            tags={"output"},
        )


@treeano.register_node("translation_and_scale_spatial_transformer")
class TranslationAndScaleSpatialTransformerNode(treeano.Wrapper1NodeImpl):

    input_keys = ("default", "final_child_output")
    hyperparameter_names = ("output_shape",)

    def compute_output(self, network, in_vw, theta_vw):
        output_shape = network.find_hyperparameter(["output_shape"])
        assert len(output_shape) == 2

        # create a matrix to convert 3 input parameters into the 6
        # parameters for an affine transform
        translation_and_scale_to_affine = np.zeros((3, 2, 3), dtype=fX)
        # first parameter is scaling
        translation_and_scale_to_affine[0, 0, 0] = 1
        translation_and_scale_to_affine[0, 1, 1] = 1
        # second parameter is x translation
        translation_and_scale_to_affine[1, 0, 2] = 1
        # third parameter is y translation
        translation_and_scale_to_affine[2, 1, 2] = 1

        theta = theta_vw.variable.dot(
            translation_and_scale_to_affine.reshape(3, 6))
        # FIXME copy-pasted from above - refactor to avoid duplication
        # calculate grid in homogeneous coordinates (x[t],y[t],1)
        grid = target_grid(output_shape)
        # calculate affine transform matrix for each element in the batch
        affine_parameters = affine_matrix_batch(theta)
        # map target coords to source coords: x[t],y[t] -> x[s],y[s]
        x_s, y_s = affine_warp_coordinates(affine_parameters, grid)
        # get new image
        out_var = warp_bilinear_interpolation(in_vw.variable,
                                              x_s,
                                              y_s,
                                              *output_shape)

        network.create_vw(
            "affine_parameters",
            variable=affine_parameters,
            shape=(in_vw.shape[0], 2, 3),
            tags={},
        )
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape[:2] + output_shape,
            tags={"output"},
        )


@treeano.register_node("rotate_shear_stretch_spatial_transformer")
class RotateShearStretchSpatialTransformerNode(treeano.Wrapper1NodeImpl):

    input_keys = ("default", "final_child_output")
    hyperparameter_names = ("output_shape",)

    def compute_output(self, network, in_vw, theta_vw):
        output_shape = network.find_hyperparameter(["output_shape"])
        assert len(output_shape) == 2

        # create a matrix to convert 3 input parameters into the 6
        # parameters for an affine transform
        # FIXME this is incorrect, and does allow scaling
        assert False
        params_to_affine = np.zeros((3, 2, 3), dtype=fX)
        # for parameters x, y, and z, convert to the following matrix:
        # x y 0
        # z 0 0
        # paramater 0 becomes top-left corner
        params_to_affine[0, 0, 0] = 1
        # parameter 1 becomes bottom-left corner
        params_to_affine[1, 1, 0] = 1
        # parameter 2 becomes top-middle
        params_to_affine[2, 0, 1] = 1

        theta = theta_vw.variable.dot(params_to_affine.reshape(3, 6))
        # FIXME copy-pasted from above - refactor to avoid duplication
        # calculate grid in homogeneous coordinates (x[t],y[t],1)
        grid = target_grid(output_shape)
        # calculate affine transform matrix for each element in the batch
        affine_parameters = affine_matrix_batch(theta)
        # map target coords to source coords: x[t],y[t] -> x[s],y[s]
        x_s, y_s = affine_warp_coordinates(affine_parameters, grid)
        # get new image
        out_var = warp_bilinear_interpolation(in_vw.variable,
                                              x_s,
                                              y_s,
                                              *output_shape)

        network.create_vw(
            "affine_parameters",
            variable=affine_parameters,
            shape=(in_vw.shape[0], 2, 3),
            tags={},
        )
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape[:2] + output_shape,
            tags={"output"},
        )


class MonitorAffineParameters(canopy.handlers.NetworkHandlerImpl):

    """
    handler that monitors the affine parameters of an
    AffineSpatialTransformerNode's localization network
    """

    def __init__(self,
                 node_name,
                 fmt="affine_parameters_%s_%s_%s",
                 input_key=None):
        """
        node_name:
        name of the affine spatial transformer node

        input_key:
        key to store the params with in the result dict

        fmt:
        format for the names of the monitor variables
        """
        self.node_name = node_name
        if input_key is None:
            input_key = "affine_parameters_%s" % node_name
        self.input_key = input_key
        self.fmt = fmt

    def transform_compile_function_kwargs(self, state, **kwargs):
        assert self.input_key not in kwargs["outputs"]
        kwargs["outputs"][self.input_key] = (self.node_name,
                                             "affine_parameters")
        return kwargs

    def call(self, fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        # batch size x 2 x 3 tensor
        affine_parameters = res.pop(self.input_key)

        at = skimage.transform.AffineTransform()
        rotation = []
        shear = []
        translation0 = []
        translation1 = []
        scale0 = []
        scale1 = []
        for m in affine_parameters:
            at.params[:2, :3] = m
            rotation.append(at.rotation)
            shear.append(at.shear)
            translation0.append(at.translation[0])
            translation1.append(at.translation[1])
            scale0.append(at.scale[0])
            scale1.append(at.scale[1])

        for param_name, param_list in [
                ("rotation", rotation),
                ("shear", shear),
                # FIXME check if this corresponds to x and y
                ("x_translation", translation0),
                ("y_translation", translation1),
                ("x_scale", scale0),
                ("y_scale", scale1),
        ]:
            def key_fn(metric):
                return self.fmt % (self.node_name, param_name, metric)

            p = np.array(param_list)
            res[key_fn("mean")] = p.mean()
            res[key_fn("std")] = p.std()
            # abs_p = np.abs(p)
            # res[key_fn("abs->mean")] = abs_p.mean()
        return res

monitor_affine_parameters = MonitorAffineParameters
