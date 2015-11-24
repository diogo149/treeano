import treeano
import theano
import theano.tensor as T
import theano.sandbox.cuda.dnn as dnn


@treeano.register_node("deconv_upsample_2d")
class DeconvUpsample2DNode(treeano.NodeImpl):

    hyperparameter_names = ('num_filters',
                            'filter_size',
                            'upsample_factor')

    def compute_output(self, network, in_vw):
        num_filters = network.find_hyperparameter(['num_filters'])
        stride = network.find_hyperparameter(['upsample_factor'])
        filter_size = network.find_hyperparameter(['filter_size'])
        pad_name = 'same'

        pad = treeano.nodes.conv.conv_parse_pad(filter_size, pad_name)
        # In the case, the 0th element of shape is the number of channels
        # in the low-res layer, and the 1st element is that of the hi-res
        # layer.  We put it in W this way, because W is a convolution from
        # hi-res to low-res.
        W = network.create_vw(
            name='weight',
            is_shared=True,
            shape=(in_vw.shape[1], num_filters,) + filter_size,
            tags={'parameter', 'weight'},
            default_inits=[],
        ).variable

        out_shape = list(in_vw.shape)
        symbolic_shape = list(in_vw.symbolic_shape())
        out_shape[1] = num_filters
        symbolic_shape[1] = num_filters
        for axis, s in zip((2, 3), stride):
            if out_shape[axis] is not None:
                out_shape[axis] *= s
            symbolic_shape[axis] *= s
        out_shape = tuple(out_shape)
        symbolic_shape = tuple(symbolic_shape)

        x = T.zeros(symbolic_shape)
        conved = dnn.dnn_conv(img=x,
                              kerns=W,
                              border_mode=pad,
                              subsample=stride)

        out_var = T.grad(None, wrt=x, known_grads={conved: in_vw.variable})

        network.create_vw(
            'default',
            variable=out_var,
            shape=out_shape,
            tags={'output'}
        )


@treeano.register_node("deconv_upsample_3d")
class DeconvUpsample3DNode(treeano.NodeImpl):

    hyperparameter_names = ('num_filters',
                            'filter_size',
                            'upsample_factor')

    def compute_output(self, network, in_vw):
        num_filters = network.find_hyperparameter(['num_filters'])
        stride = network.find_hyperparameter(['upsample_factor'])
        filter_size = network.find_hyperparameter(['filter_size'])
        pad_name = 'same'

        pad = treeano.nodes.conv.conv_parse_pad(filter_size, pad_name)
        # In the case, the 0th element of shape is the number of channels
        # in the low-res layer, and the 1st element is that of the hi-res
        # layer.  We put it in W this way, because W is a convolution from
        # hi-res to low-res.
        W = network.create_vw(
            name='weight',
            is_shared=True,
            shape=(in_vw.shape[1], num_filters,) + filter_size,
            tags={'parameter', 'weight'},
            default_inits=[],
        ).variable

        out_shape = list(in_vw.shape)
        symbolic_shape = list(in_vw.symbolic_shape())
        out_shape[1] = num_filters
        symbolic_shape[1] = num_filters
        for axis, s in zip((2, 3, 4), stride):
            if out_shape[axis] is not None:
                out_shape[axis] *= s
            symbolic_shape[axis] *= s
        out_shape = tuple(out_shape)
        symbolic_shape = tuple(symbolic_shape)

        x = T.zeros(symbolic_shape)
        conved = dnn.dnn_conv3d(img=x,
                                kerns=W,
                                border_mode=pad,
                                subsample=stride)

        out_var = T.grad(None, wrt=x, known_grads={conved: in_vw.variable})

        network.create_vw(
            'default',
            variable=out_var,
            shape=out_shape,
            tags={'output'}
        )
