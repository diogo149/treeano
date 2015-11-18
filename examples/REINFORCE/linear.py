from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import REINFORCE

fX = theano.config.floatX


TARGET_WEIGHT = np.random.randn(10, 2).astype(fX)
TARGET_BIAS = np.random.randn(2).astype(fX)


class RewardNode(treeano.NodeImpl):

    input_keys = ("state", "sampled")

    def compute_output(self, network, state_vw, sampled_vw):
        W = T.constant(TARGET_WEIGHT)
        b = T.constant(TARGET_BIAS)
        target = T.dot(state_vw.variable, W) + b.dimshuffle("x", 0)
        reward = -T.sqr(sampled_vw.variable - target).sum(axis=1)
        network.create_vw(
            "raw_reward",
            variable=T.mean(reward),
            shape=(),
        )
        baseline_reward = 100
        network.create_vw(
            "default",
            variable=reward + baseline_reward,
            shape=(state_vw.shape[0],),
            tags={"output"},
        )


BATCH_SIZE = 64
graph = tn.GraphNode(
    "graph",
    [[tn.InputNode("state", shape=(BATCH_SIZE, 10)),
      tn.DenseNode("mu", num_units=2),
      tn.ConstantNode("sigma", value=1.),
      REINFORCE.NormalSampleNode("sampled"),
      RewardNode("reward"),
      REINFORCE.NormalREINFORCECostNode("REINFORCE")],
     [{"from": "state", "to": "mu"},
      {"from": "mu", "to": "sampled", "to_key": "mu"},
      {"from": "sigma", "to": "sampled", "to_key": "sigma"},
      {"from": "sampled", "to": "reward", "to_key": "sampled"},
      {"from": "state", "to": "reward", "to_key": "state"},
      {"from": "state", "to": "REINFORCE", "to_key": "state"},
      {"from": "mu", "to": "REINFORCE", "to_key": "mu"},
      {"from": "sigma", "to": "REINFORCE", "to_key": "sigma"},
      {"from": "reward", "to": "REINFORCE", "to_key": "reward"},
      {"from": "sampled", "to": "REINFORCE", "to_key": "sampled"},
      {"from": "REINFORCE"}]]
)

network = tn.AdamNode(
    "adam",
    {"subtree": graph,
     "cost": tn.ReferenceNode("cost", reference="REINFORCE")},
    learning_rate=0.1
).network()
fn = network.function(
    ["state"], [("reward", "raw_reward")], include_updates=True)

errors = []
for i in range(5000):
    error, = fn(np.random.randn(BATCH_SIZE, 10).astype(fX))
    if i % 100 == 0:
        print("Iter:", i, "Error:", error)
    errors.append(error)

print("mean reward:", np.mean(errors))
