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
import canopy
import canopy.sandbox.datasets

fX = theano.config.floatX

# ############################### prepare data ###############################

train, valid, test = canopy.sandbox.datasets.mnist()

# ############################## prepare model ##############################
# architecture:
# - fully connected 10 units
# - softmax

# - the batch size can be provided as `None` to make the network
#   work for multiple different batch sizes
model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(None, 1, 28, 28)),
         tn.DenseNode("fc3", num_units=10),
         tn.SoftmaxNode("pred"),
         ]),
    inits=[treeano.inits.XavierNormalInit()],
)

with_updates = tn.HyperparameterNode(
    "with_updates",
    tn.AdamNode(
        "adam",
        {"subtree": model,
         "cost": tn.TotalCostNode("cost", {
             "pred": tn.ReferenceNode("pred_ref", reference="model"),
             "target": tn.InputNode("y", shape=(None,), dtype="int32")},
         )}),
    cost_function=treeano.utils.categorical_crossentropy_i32,
)
network = with_updates.network()
network.build()  # build eagerly to share weights

train_fn = network.function(["x", "y"], ["cost"], include_updates=True)
valid_fn = network.function(["x", "y"], ["cost", "pred"])

# ################################# training #################################

print("Starting training...")

num_epochs = 25
batch_size = 100
for epoch_num in range(num_epochs):
    # iterate over training minibatches and update the weights
    num_batches_train = int(np.ceil(len(train["x"]) / batch_size))
    train_losses = []
    for batch_num in range(num_batches_train):
        batch_slice = slice(batch_size * batch_num,
                            batch_size * (batch_num + 1))
        X_batch = train["x"][batch_slice]
        y_batch = train["y"][batch_slice]

        loss, = train_fn(X_batch, y_batch)
        train_losses.append(loss)
    # aggregate training losses for each minibatch into scalar
    train_loss = np.mean(train_losses)

    # calculate validation loss
    num_batches_valid = int(np.ceil(len(valid["x"]) / batch_size))
    valid_losses = []
    list_of_probabilities_batch = []
    for batch_num in range(num_batches_valid):
        batch_slice = slice(batch_size * batch_num,
                            batch_size * (batch_num + 1))
        X_batch = valid["x"][batch_slice]
        y_batch = valid["y"][batch_slice]

        loss, probabilities_batch = valid_fn(X_batch, y_batch)
        valid_losses.append(loss)
        list_of_probabilities_batch.append(probabilities_batch)
    valid_loss = np.mean(valid_losses)
    # concatenate probabilities for each batch into a matrix
    probabilities = np.concatenate(list_of_probabilities_batch)
    # calculate classes from the probabilities
    predicted_classes = np.argmax(probabilities, axis=1)
    # calculate accuracy for this epoch
    accuracy = sklearn.metrics.accuracy_score(valid["y"], predicted_classes)

    print("Epoch: %d, train_loss=%f, valid_loss=%f, valid_accuracy=%f"
          % (epoch_num + 1, train_loss, valid_loss, accuracy))
