import nose.tools as nt
from treeano import core
import treeano
import treeano.nodes as tn


def test_list_children_container():
    # test to_data
    cc = core.ListChildrenContainer([])
    nt.assert_equal([], cc.to_data())

    # test children_container_to_data
    as_data = core.children_container_to_data(cc)
    nt.assert_equal(dict(
        children_container_key="list",
        children_container_data=[],
    ),
        as_data)

    # test back and forth
    cc2 = core.children_container_from_data(as_data)
    nt.assert_is_instance(cc2, core.ListChildrenContainer)
    nt.assert_equal(as_data,
                    core.children_container_to_data(cc2))


def test_dict_children_container():
    # test to_data
    cc = core.DictChildrenContainer({})
    nt.assert_equal({}, cc.to_data())

    # test children_container_to_data
    as_data = core.children_container_to_data(cc)
    nt.assert_equal(dict(
        children_container_key="dict",
        children_container_data={},
    ),
        as_data)

    # test back and forth
    cc2 = core.children_container_from_data(as_data)
    nt.assert_is_instance(cc2, core.DictChildrenContainer)
    nt.assert_equal(as_data,
                    core.children_container_to_data(cc2))


def test_dict_children_container_schema():
    dccs = core.DictChildrenContainerSchema(
        foo=core.ListChildrenContainer,
        bar=core.ChildContainer,
    )
    node = tn.AddConstantNode("hello")
    cc1 = dccs({"foo": [node, node], "bar": node})
    cc2 = core.children_container._DictChildrenContainerFromSchema(
        {"foo": core.ListChildrenContainer([node, node]),
         "bar": core.ChildContainer(node)})
    # test that it makes the expected class
    nt.assert_equal(cc1, cc2)


def test_dict_children_container_schema_children():
    dccs = core.DictChildrenContainerSchema(
        foo=core.ListChildrenContainer,
        bar=core.ChildContainer,
    )
    node = tn.AddConstantNode("hello")
    in_map = {"foo": [node, node], "bar": node}
    cc = dccs(in_map)
    # test that .children returns the same as the input
    nt.assert_equal(cc.children, in_map)


def test_dict_children_container_schema_serialization():
    dccs = core.DictChildrenContainerSchema(
        foo=core.ListChildrenContainer,
        bar=core.ChildContainer,
    )
    node = tn.AddConstantNode("hello")
    in_map = {"foo": [node, node], "bar": node}
    cc1 = dccs(in_map)
    cc2 = core.children_container_from_data(
        core.children_container_to_data(cc1))

    nt.assert_equal(cc1.__class__, cc2.__class__)
    nt.assert_equal(cc1.__dict__, cc2.__dict__)


def test_dict_children_container_schema_optional_children():
    dccs = core.DictChildrenContainerSchema(
        foo=core.ListChildrenContainer,
        bar=core.ChildContainer,
    )
    node = tn.AddConstantNode("hello")
    in_map = {"foo": [node, node]}
    cc = dccs(in_map)
    # test that .children returns the same as the input
    nt.assert_equal(cc.children, in_map)


def test_dict_children_container_schema_no_children():
    dccs = core.DictChildrenContainerSchema(
        foo=core.ListChildrenContainer,
        bar=core.ChildContainer,
    )
    cc = dccs(None)
    # test that .children returns the same as the input
    nt.assert_equal(cc.children, {})


def test_nodes_and_edges_children_container():
    # test to_data
    cc = core.NodesAndEdgesContainer([[], []])
    nt.assert_equal({"nodes": [], "edges": []}, cc.to_data())

    # test children_container_to_data
    as_data = core.children_container_to_data(cc)
    nt.assert_equal(dict(
        children_container_key="nodes_and_edges",
        children_container_data={"nodes": [], "edges": []}),
        as_data)

    # test back and forth
    cc2 = core.children_container_from_data(as_data)
    nt.assert_is_instance(cc2, core.NodesAndEdgesContainer)
    nt.assert_equal(as_data,
                    core.children_container_to_data(cc2))
