"""
NOTE: concatenation seems very slow
"""

import treeano
import treeano.nodes as tn


@treeano.register_node("partition_axis")
class PartitionAxisNode(treeano.NodeImpl):

    """
    node that returns a fraction of the input tensor

    rough explanation:
    x.shape == (4, 8, 12, 16, 20)
    y = partition_axis(x, split_idx=2, num_splits=4, channel_axis=3)
    =>
    y == x[:, :, :, 8:12, :]
    """

    hyperparameter_names = ("split_idx",
                            "num_splits",
                            "channel_axis")

    def compute_output(self, network, in_vw):
        # FIXME make default in terms of batch axis
        channel_axis = network.find_hyperparameter(["channel_axis"], 1)
        split_idx = network.find_hyperparameter(["split_idx"])
        num_splits = network.find_hyperparameter(["num_splits"])

        var = in_vw.variable
        shape = in_vw.shape

        num_channels = shape[channel_axis]
        start_idx = (num_channels * split_idx) // num_splits
        end_idx = num_channels * (split_idx + 1) // num_splits

        new_shape = list(shape)
        new_shape[channel_axis] = end_idx - start_idx
        new_shape = tuple(new_shape)

        idx = tuple([slice(None) for _ in range(channel_axis)]
                    + [slice(start_idx, end_idx)])
        network.create_vw(
            "default",
            variable=var[idx],
            shape=new_shape,
            tags={"output"},
        )


def MultiPool2DNode(name, **kwargs):
    # TODO tests
    # TODO make a node that verifies hyperparameters
    return tn.HyperparameterNode(
        name,
        tn.ConcatenateNode(
            name + "_concat",
            [tn.SequentialNode(name + "_seq0",
                               [PartitionAxisNode(name + "_part0",
                                                  split_idx=0,
                                                  num_splits=2),
                                tn.MaxPool2DNode(name + "_max",
                                                 ignore_border=True)]),
             tn.SequentialNode(name + "_seq1",
                               [PartitionAxisNode(name + "_part1",
                                                  split_idx=1,
                                                  num_splits=2),
                                tn.MeanPool2DNode(name + "_mean")])]),
        **kwargs)
