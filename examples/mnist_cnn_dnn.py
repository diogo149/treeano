from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import sklearn.datasets
import sklearn.cross_validation
import sklearn.metrics
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
import treeano.lasagne.nodes as tl
import canopy

fX = theano.config.floatX

# ############################### prepare data ###############################

mnist = sklearn.datasets.fetch_mldata('MNIST original')
# theano has a constant float type that it uses (float32 for GPU)
# also rescaling to [0, 1] instead of [0, 255]
X = mnist['data'].reshape(-1, 1, 28, 28).astype(fX) / 255.0
y = mnist['target'].astype("int32")
X_train, X_valid, y_train, y_valid = sklearn.cross_validation.train_test_split(
    X, y, random_state=42)

# ############################## prepare model ##############################
# architecture:
# - 5x5 conv, 32 filters
# - 2x2 maxpool
# - ReLU
# - 5x5 conv, 32 filters
# - 2x2 maxpool
# - ReLU
# - fully connected layer - 256 units
# - 50% dropout
# - fully connected layer- 10 units
# - softmax

# - the batch size can be provided as `None` to make the network
#   work for multiple different batch sizes
model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(None, 1, 28, 28)),
         tl.Conv2DDNNNode("conv1"),
         tl.MaxPool2DDNNNode("mp1"),
         tn.ReLUNode("relu1"),
         tl.Conv2DDNNNode("conv2"),
         tn.ReLUNode("relu2"),
         tl.MaxPool2DDNNNode("mp2"),
         tn.DenseNode("fc1"),
         tn.DropoutNode("do1"),
         tn.DenseNode("fc2", num_units=10),
         tn.SoftmaxNode("pred"),
         ]),
    num_filters=32,
    filter_size=(5, 5),
    pool_size=(2, 2),
    num_units=256,
    dropout_probability=0.5,
    inits=[treeano.inits.XavierNormalInit()],
)

with_updates = tn.HyperparameterNode(
    "with_updates",
    tn.AdamNode(
        "adam",
        {"subtree": model,
         "cost": tn.PredictionCostNode("cost", {
             "pred": tn.ReferenceNode("pred_ref", reference="model"),
             "target": tn.InputNode("y", shape=(None,), dtype="int32")},
         )}),
    loss_function=treeano.utils.categorical_crossentropy_i32,
)
network = with_updates.network()
network.build()  # build eagerly to share weights

train_fn = canopy.handled_function(
    network,
    [canopy.handlers.chunk_variables(batch_size=100, variables=["x", "y"])],
    ["x", "y"],
    ["cost"],
    include_updates=True)

valid_fn = canopy.handled_function(
    network,
    [canopy.handlers.override_hyperparameters(dropout_probability=0),
     canopy.handlers.chunk_variables(batch_size=100, variables=["x", "y"])],
    ["x", "y"],
    ["cost", "pred"])


# ################################# training #################################

print("Starting training...")

num_epochs = 25
batch_size = 100
for epoch_num in range(num_epochs):
    train_loss, = train_fn(X_train, y_train)
    valid_loss, probabilities = valid_fn(X_valid, y_valid)
    predicted_classes = np.argmax(probabilities, axis=1)
    # calculate accuracy for this epoch
    accuracy = sklearn.metrics.accuracy_score(y_valid, predicted_classes)
    print("Epoch: %d, train_loss=%f, valid_loss=%f, valid_accuracy=%f"
          % (epoch_num + 1, train_loss, valid_loss, accuracy))
