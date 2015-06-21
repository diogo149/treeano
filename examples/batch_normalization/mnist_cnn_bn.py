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

from treeano.sandbox.nodes import batch_normalization as bn

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
model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(None, 1, 28, 28)),
         tl.Conv2DNode("conv1"),
         bn.BatchNormalizationNode("bn1"),
         tl.MaxPool2DNode("mp1"),
         tn.ReLUNode("relu1"),
         tl.Conv2DNode("conv2"),
         bn.BatchNormalizationNode("bn2"),
         tn.ReLUNode("relu2"),
         tl.MaxPool2DNode("mp2"),
         tn.DenseNode("fc1"),
         bn.BatchNormalizationNode("bn3"),
         tn.DenseNode("fc2", num_units=10),
         bn.BatchNormalizationNode("bn4"),
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

train_fn = network.function(["x", "y"], ["cost"], include_updates=True)

# different ways of disabling dropout
if False:
    network_no_dropout = canopy.transforms.remove_dropout(network)
    valid_fn = network_no_dropout.function(["x", "y"], ["cost", "pred"])
else:
    valid_fn = canopy.handled_function(
        network,
        [canopy.handlers.override_hyperparameters(dropout_probability=0)],
        ["x", "y"],
        ["cost", "pred"])


# ################################# training #################################

print("Starting training...")

num_epochs = 25
batch_size = 100
for epoch_num in range(num_epochs):
    # iterate over training minibatches and update the weights
    num_batches_train = int(np.ceil(len(X_train) / batch_size))
    train_losses = []
    for batch_num in range(num_batches_train):
        batch_slice = slice(batch_size * batch_num,
                            batch_size * (batch_num + 1))
        X_batch = X_train[batch_slice]
        y_batch = y_train[batch_slice]

        loss, = train_fn(X_batch, y_batch)
        train_losses.append(loss)
    # aggregate training losses for each minibatch into scalar
    train_loss = np.mean(train_losses)

    # calculate validation loss
    num_batches_valid = int(np.ceil(len(X_valid) / batch_size))
    valid_losses = []
    list_of_probabilities_batch = []
    for batch_num in range(num_batches_valid):
        batch_slice = slice(batch_size * batch_num,
                            batch_size * (batch_num + 1))
        X_batch = X_valid[batch_slice]
        y_batch = y_valid[batch_slice]

        loss, probabilities_batch = valid_fn(X_batch, y_batch)
        valid_losses.append(loss)
        list_of_probabilities_batch.append(probabilities_batch)
    valid_loss = np.mean(valid_losses)
    # concatenate probabilities for each batch into a matrix
    probabilities = np.concatenate(list_of_probabilities_batch)
    # calculate classes from the probabilities
    predicted_classes = np.argmax(probabilities, axis=1)
    # calculate accuracy for this epoch
    accuracy = sklearn.metrics.accuracy_score(y_valid, predicted_classes)

    print("Epoch: %d, train_loss=%f, valid_loss=%f, valid_accuracy=%f"
          % (epoch_num + 1, train_loss, valid_loss, accuracy))
