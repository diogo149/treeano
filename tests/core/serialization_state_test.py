import nose.tools as nt
from treeano import core


@nt.raises(AssertionError)
def test_duplicate_register_child_container():
    @core.register_children_container("list")
    class Foo(object):
        pass


@nt.raises(AssertionError)
def test_duplicate_register_node():
    @core.register_node("input")
    class Foo(object):
        pass
