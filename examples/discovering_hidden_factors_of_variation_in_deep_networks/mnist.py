from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import itertools
import numpy as np
import sklearn.datasets
import sklearn.cross_validation
import sklearn.metrics
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
from treeano.sandbox.utils import cross_covariance
import canopy
import matplotlib.pyplot as plt

fX = theano.config.floatX

LATENT_SIZE = 2


# ############################### prepare data ###############################

mnist = sklearn.datasets.fetch_mldata('MNIST original')
# theano has a constant float type that it uses (float32 for GPU)
# also rescaling to [0, 1] instead of [0, 255]
X = mnist['data'].astype(fX) / 255.0
y = mnist['target'].astype("int32")
X_train, X_valid, y_train, y_valid = sklearn.cross_validation.train_test_split(
    X, y, random_state=42)
in_train = {"x": X_train, "y": y_train}
in_valid = {"x": X_valid, "y": y_valid}

# ############################## prepare model ##############################
# architecture:
# - fully connected 512 units
# - ReLU
# - 50% dropout
# - fully connected 512 units
# - ReLU
# - 50% dropout
# - fully connected 10 units
# - softmax

# - the batch size can be provided as `None` to make the network
#   work for multiple different batch sizes
model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(None, 28 * 28)),
         tn.DenseNode("fc1"),
         tn.ReLUNode("relu1"),
         # tn.DropoutNode("do1"),
         tn.DenseNode("fc2"),
         tn.ReLUNode("relu2"),
         # tn.DropoutNode("do2"),
         tn.ConcatenateNode(
             "concat",
             [tn.SequentialNode(
                 "y_vars",
                 [tn.DenseNode("fc_y", num_units=10),
                  tn.SoftmaxNode("y_pred"),
                  tn.AuxiliaryCostNode(
                      "classification_cost",
                      {"target": tn.InputNode("y",
                                              shape=(None,),
                                              dtype="int32")},
                      cost_function=treeano.utils.categorical_crossentropy_i32)]),
              tn.SequentialNode(
                  "z_vars",
                  [tn.DenseNode("fc_z", num_units=LATENT_SIZE),
                   tn.AuxiliaryCostNode(
                       "xcov_cost",
                       {"target": tn.ReferenceNode("y_ref",
                                                   reference="y_pred")},
                       cost_function=cross_covariance)])],
             axis=1),
         tn.DenseNode("fc3"),
         tn.ReLUNode("relu3"),
         tn.DenseNode("fc4"),
         tn.ReLUNode("relu4"),
         tn.DenseNode("reconstruction", num_units=28 * 28),
         tn.TotalCostNode(
             "cost",
             {"pred": tn.IdentityNode("recon_id"),
              "target": tn.ReferenceNode("in_ref", reference="x")},
             cost_function=treeano.utils.squared_error),
         tn.MultiplyConstantNode("mul_reconstruction_error", value=0.1),
         tn.InputElementwiseSumNode("total_cost")]),
    num_units=512,
    cost_reference="total_cost",
    dropout_probability=0.5,
    inits=[treeano.inits.XavierNormalInit()],
)

with_updates = tn.HyperparameterNode(
    "with_updates",
    tn.AdamNode(
        "adam",
        {"subtree": model,
         "cost": tn.ReferenceNode("cost_ref", reference="total_cost")}),
)
network = with_updates.network()
network.build()  # build eagerly to share weights

BATCH_SIZE = 500

valid_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="valid_time"),
     canopy.handlers.override_hyperparameters(dropout_probability=0),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"total_cost": "total_cost",
     "pred": "y_pred"})


def validate(in_dict, results_dict):
    valid_out = valid_fn(in_valid)
    probabilities = valid_out["pred"]
    predicted_classes = np.argmax(probabilities, axis=1)
    results_dict["valid_total_cost"] = valid_out["total_cost"]
    results_dict["valid_time"] = valid_out["valid_time"]
    results_dict["valid_accuracy"] = sklearn.metrics.accuracy_score(
        y_valid, predicted_classes)

train_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="total_time"),
     canopy.handlers.call_after_every(1, validate),
     canopy.handlers.time_call(key="train_time"),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"train_cost": "total_cost"},
    include_updates=True)


# ################################# training #################################

print("Starting training...")
canopy.evaluate_until(fn=train_fn,
                      gen=itertools.repeat(in_train),
                      max_iters=25)

# ############################## reconstruction ##############################

reconstruction_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="time"),
     canopy.handlers.override_hyperparameters(dropout_probability=0),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x"])],
    {"x": "x"},
    {"reconstruction": "reconstruction",
     "y": "y_pred",
     "z": "fc_z"})

recon = reconstruction_fn(in_train)

idx = 149
plt.imshow(in_train["x"][idx].reshape(28, 28), cmap=plt.cm.gray)
plt.show()
plt.imshow(recon["reconstruction"][idx].reshape(28, 28), cmap=plt.cm.gray)
plt.show()

# ####################### playing with hidden factors #######################

y = T.ivector()
hallucinate_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="time"),
     canopy.handlers.override_hyperparameters(dropout_probability=0)],
    {"y": y,
     "z": "fc_z"},
    {"reconstruction": "reconstruction"},
    givens={"y_pred": T.extra_ops.to_one_hot(y, 10)}
)


def plot_z_range(z_idx, z_min=-0.5, z_max=0.5, num_variations=9):
    ys = np.repeat(np.arange(10), num_variations).astype('int32')
    z_vals = np.tile(np.linspace(z_min, z_max, num_variations), 10).astype(fX)
    z_zeros = np.zeros_like(z_vals)
    zs = np.vstack([z_zeros] * z_idx
                   + [z_vals]
                   + [z_zeros] * (LATENT_SIZE - z_idx - 1)).T
    reconstruction = hallucinate_fn({"y": ys, "z": zs})["reconstruction"]
    img = reconstruction.reshape(
        10, num_variations, 28, 28
    ).transpose(
        1, 2, 0, 3
    ).reshape(num_variations * 28, 10 * 28)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    return ys, zs, reconstruction

ys, zs, reconstruction = plot_z_range(0)

num_hallucinate = 100
ys = np.random.randint(0, 10, num_hallucinate).astype(np.int32)
zs = np.random.randint()
res = hallucinate_fn({"y": [9], "z": [[0., 0.]]})["reconstruction"]
plt.imshow(res["reconstruction"][idx].reshape(28, 28), cmap=plt.cm.gray)
plt.show()
