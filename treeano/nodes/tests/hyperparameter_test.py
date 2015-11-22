import copy

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_variable_hyperparameter_node_serialization():
    tn.check_serialization(tn.VariableHyperparameterNode("a",
                                                         tn.IdentityNode("b")))


def test_output_hyperparameter_node_serialization():
    tn.check_serialization(tn.OutputHyperparameterNode("a"))


def test_variable_hyperparameter_node():
    network = tn.VariableHyperparameterNode(
        "a",
        tn.InputNode("b", shape=())).network()
    hp = network["a"].get_vw("hyperparameter").variable
    nt.assert_equal(hp.ndim, 0)
    fn = network.function([("a", "hyperparameter")], [hp])
    x = 42
    nt.assert_equal(fn(x), [x])


def test_shared_hyperparameter_node():
    network = tn.SharedHyperparameterNode(
        "a",
        tn.InputNode("b", shape=())).network()
    hp = network["a"].get_vw("hyperparameter").variable
    nt.assert_equal(hp.ndim, 0)
    fn1 = network.function([("a", "hyperparameter")],
                           [hp],
                           include_updates=True)
    fn2 = network.function([], [hp])
    x = 42
    nt.assert_equal(fn1(x), [x])
    nt.assert_equal(fn2(), [x])


def test_output_hyperparameter_node():
    network = tn.VariableHyperparameterNode(
        "a",
        tn.OutputHyperparameterNode("b"),
        hyperparameter="foobar"
    ).network()
    fn = network.function([("a", "hyperparameter")], ["b"])
    x = 253
    nt.assert_equal(fn(x), [x])


def test_variable_hyperparameter_node_double():
    network = tn.VariableHyperparameterNode(
        "a",
        tn.VariableHyperparameterNode(
            "b",
            tn.OutputHyperparameterNode("c", hyperparameter="foo"),
            hyperparameter="bar"),
        hyperparameter="foo").network()
    fn = network.function([("a", "hyperparameter")], ["c"])
    x = 253
    nt.assert_equal(fn(x), [x])
