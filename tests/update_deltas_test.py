import theano
import treeano


def test_update_deltas():
    x = theano.shared(0, name="x")
    ud = treeano.UpdateDeltas({x: 0})
    ud += 1
    ud *= 2
    fn = theano.function([], updates=ud.to_updates())
    fn()
    assert x.get_value() == 2
    fn()
    assert x.get_value() == 4


def test_update_deltas_getitem():
    x = theano.shared(0, name="x")
    ud = treeano.UpdateDeltas({})
    assert ud[x] == 0
    ud = treeano.UpdateDeltas({x: 5})
    fn = theano.function([], updates=ud.to_updates())
    fn()
    assert x.get_value() == 5
    fn()
    assert x.get_value() == 10


def test_update_deltas_setitem():
    x = theano.shared(0, name="x")
    ud = treeano.UpdateDeltas({})
    ud[x] += 3
    assert ud[x] == 3
    ud[x] = 7
    assert ud[x] == 7
    fn = theano.function([], updates=ud.to_updates())
    fn()
    assert x.get_value() == 7


def test_update_deltas_add1():
    x = theano.shared(0, name="x")
    ud1 = treeano.UpdateDeltas({x: 3})
    ud1b = ud1
    ud2 = treeano.UpdateDeltas({x: 4})
    ud3 = ud1 + ud2
    assert ud1[x] == 3
    assert ud1b[x] == 3
    assert ud2[x] == 4
    assert ud3[x] == 7


def test_update_deltas_iadd1():
    x = theano.shared(0, name="x")
    ud1 = treeano.UpdateDeltas({x: 3})
    ud1b = ud1
    ud2 = treeano.UpdateDeltas({x: 4})
    ud1 += ud2
    assert ud1[x] == 7
    assert ud1b[x] == 7
    assert ud2[x] == 4


def test_update_deltas_mul1():
    x = theano.shared(0, name="x")
    ud1 = treeano.UpdateDeltas({x: 3})
    ud2 = ud1
    ud1 = ud1 * 2
    assert ud1[x] == 6
    assert ud2[x] == 3


def test_update_deltas_imul1():
    x = theano.shared(0, name="x")
    ud1 = treeano.UpdateDeltas({x: 3})
    ud2 = ud1
    ud1 *= 2
    assert ud1[x] == 6
    assert ud2[x] == 6
