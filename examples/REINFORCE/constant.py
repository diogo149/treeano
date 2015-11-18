import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import REINFORCE

fX = theano.config.floatX


class ConstantStateNode(treeano.NodeImpl):

    input_keys = ()
    hyperparameter_names = ("shape",)

    def compute_output(self, network):
        shape = network.find_hyperparameter(["shape"])
        network.create_vw(
            "default",
            is_shared=True,
            shape=shape,
            tags={"parameter"},
            default_inits=[],
        )


def reward_fn(x):
    return -T.sqr(x - 3.5).sum(axis=1) + 100

graph = tn.GraphNode(
    "graph",
    [[tn.ConstantNode("state", value=T.zeros((1, 1))),
      ConstantStateNode("mu", shape=(1, 1)),
      tn.ConstantNode("sigma", value=1.),
      REINFORCE.NormalSampleNode("sampled"),
      tn.ApplyNode("reward", fn=reward_fn, shape_fn=lambda x: x[:1]),
      REINFORCE.NormalREINFORCECostNode("REINFORCE")],
     [{"from": "mu", "to": "sampled", "to_key": "mu"},
      {"from": "sigma", "to": "sampled", "to_key": "sigma"},
      {"from": "sampled", "to": "reward"},
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
fn = network.function([], ["graph", "mu"], include_updates=True)

mus = []
for i in range(1000):
    _, mu = fn()
    print("Iter:", i, "Predicted constant:", mu)
    mus.append(mu)

print("MSE from optimal constant:", np.mean((np.array(mus) - 3.5) ** 2))
