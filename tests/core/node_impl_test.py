import nose.tools as nt
from treeano import core


@nt.raises(AssertionError)
def test_node_impl_hyperparameter_names():
    class FooNode(core.NodeImpl):
        hyperparameter_names = ("a", "b")

    FooNode(name="foo", c=3)


def test_node_impl_repr1():
    class FooNode(core.NodeImpl):
        hyperparameter_names = ("a", "b")

    nt.assert_equal(repr(FooNode(name="foo", a=3)),
                    "FooNode(name='foo', a=3)")

    nt.assert_equal(repr(FooNode(name="foo", a=3)),
                    str(FooNode(name="foo", a=3)),)


def test_node_impl_repr_children_container():
    class FooNode(core.NodeImpl):
        hyperparameter_names = ("a", "b")
        children_container = core.ListChildrenContainer

    node = FooNode(name="foo",
                   a=3,
                   children=[FooNode(name="bar1",
                                     children=[FooNode(name="choo",
                                                       children=[])]),
                             FooNode(name="bar2",
                                     children=[]), ])
    nt.assert_equal(repr(node),
                    """
FooNode(name='foo', a=3)
| FooNode(name='bar1')
| | FooNode(name='choo')
| FooNode(name='bar2')
""".strip())


def test_node_impl_get_hyperparameter1():
    class FooNode(core.NodeImpl):
        hyperparameter_names = ("a", "b")

    nt.assert_equal(FooNode(name="foo", a=3).get_hyperparameter(None, "a"), 3)


@nt.raises(core.MissingHyperparameter)
def test_node_impl_get_hyperparameter2():
    class FooNode(core.NodeImpl):
        hyperparameter_names = ("a", "b")

    FooNode(name="foo", a=3).get_hyperparameter(None, "b")
