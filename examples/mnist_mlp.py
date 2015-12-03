from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import time
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
        [tn.InputNode("x", shape=(None, 1, 28, 28)),
         tn.DenseNode("fc1"),
         tn.ReLUNode("relu1"),
         tn.DropoutNode("do1"),
         tn.DenseNode("fc2"),
         tn.ReLUNode("relu2"),
         tn.DropoutNode("do2"),
         tn.DenseNode("fc3", num_units=10),
         tn.SoftmaxNode("pred"),
         ]),
    num_units=512,
    dropout_probability=0.5,
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


BATCH_SIZE = 500
train_fn = canopy.handled_fn(
    network,
    [canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"cost": "cost"},
    include_updates=True)

valid_fn = canopy.handled_fn(
    network,
    [canopy.handlers.override_hyperparameters(dropout_probability=0),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"cost": "cost", "pred": "pred"})


# ################################# training #################################

print("Starting training...")

NUM_EPOCHS = 25
for epoch_num in range(NUM_EPOCHS):
    start_time = time.time()
    train_loss = train_fn(train)["cost"]
    valid_out = valid_fn(valid)
    valid_loss, probabilities = valid_out["cost"], valid_out["pred"]
    predicted_classes = np.argmax(probabilities, axis=1)
    # calculate accuracy for this epoch
    accuracy = sklearn.metrics.accuracy_score(valid["y"], predicted_classes)
    total_time = time.time() - start_time
    print("Epoch: %d, train_loss=%f, valid_loss=%f, accuracy=%f, time=%fs"
          % (epoch_num + 1, train_loss, valid_loss, accuracy, total_time))
