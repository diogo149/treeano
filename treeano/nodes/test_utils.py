"""
some utilities for testing nodes
"""
import json

from .. import core


def assert_equal(t1, t2, msg=None):
    # TODO factor out somewhere else
    assert t1 == t2, dict(
        msg=msg,
        t1=t1,
        t2=t2,
    )


def check_serialization(node):
    # test __eq__
    assert_equal(
        node,
        node)
    # test serialization
    assert_equal(
        node,
        core.node_from_data(core.node_to_data(node)))
    # test that serialization is json-serializable
    assert_equal(
        node,
        core.node_from_data(json.loads(json.dumps(core.node_to_data(node)))))
