import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


@treeano.register_node("auxiliary_dense_softmax_categorical_crossentropy")
class AuxiliaryDenseSoftmaxCCENode(treeano.WrapperNodeImpl):

    hyperparameter_names = (tn.DenseNode.hyperparameter_names
                            + tn.AuxiliaryCostNode.hyperparameter_names)
    children_container = treeano.core.DictChildrenContainerSchema(
        target=treeano.core.ChildContainer,
    )

    def architecture_children(self):
        return [tn.AuxiliaryCostNode(
            self.name + "_auxiliary",
            {"target": self.raw_children()["target"],
             "pre_cost": tn.SequentialNode(
                 self.name + "_sequential",
                 [tn.DenseNode(self.name + "_dense"),
                  tn.SoftmaxNode(self.name + "_softmax")])},
            cost_function=T.nnet.categorical_crossentropy)]
