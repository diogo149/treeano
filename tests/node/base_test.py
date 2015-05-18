import treeano.node.base as base
import nose.tools as nt


def test_list_children_container():
    # test to_data
    cc = base.ListChildrenContainer([])
    nt.assert_equal([], cc.to_data())

    # test children_container_to_data
    as_data = base.children_container_to_data(cc)
    nt.assert_equal(dict(
        children_container_key="list",
        children_container_data=[],
    ),
        as_data)

    # test back and forth
    cc2 = base.children_container_from_data(as_data)
    nt.assert_is_instance(cc2, base.ListChildrenContainer)
    nt.assert_equal(as_data,
                    base.children_container_to_data(cc2))


@nt.raises(AssertionError)
def test_node_impl_hyperparameter_names():
    class FooNode(base.NodeImpl):
        hyperparameter_names = ("a", "b")

    FooNode(name="foo", c=3)


def test_node_impl_repr1():
    class FooNode(base.NodeImpl):
        hyperparameter_names = ("a", "b")

    nt.assert_equal(repr(FooNode(name="foo", a=3)),
                    "FooNode(name='foo', a=3)")

    nt.assert_equal(repr(FooNode(name="foo", a=3)),
                    str(FooNode(name="foo", a=3)),)


def test_node_impl_repr_children_container():
    class FooNode(base.NodeImpl):
        hyperparameter_names = ("a", "b")
        children_container = base.ListChildrenContainer

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
    class FooNode(base.NodeImpl):
        hyperparameter_names = ("a", "b")

    nt.assert_equal(FooNode(name="foo", a=3).get_hyperparameter(None, "a"), 3)


@nt.raises(base.MissingHyperparameter)
def test_node_impl_get_hyperparameter2():
    class FooNode(base.NodeImpl):
        hyperparameter_names = ("a", "b")

    FooNode(name="foo", a=3).get_hyperparameter(None, "b")
