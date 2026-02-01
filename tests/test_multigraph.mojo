from networkx import MultiGraph
from testing import assert_equal, assert_true, assert_false
from utils import Variant


comptime AttrValue = Variant[Int, Float64, Bool, String]


fn main() raises:
    var g = MultiGraph[Int]()
    var k0 = g.add_edge(1, 2, 3.0)
    var k1 = g.add_edge(1, 2, 5.0)

    assert_true(g.has_node(1))
    assert_true(g.has_node(2))

    assert_true(g.has_edge(1, 2))
    assert_true(g.has_edge_key(1, 2, k0))
    assert_true(g.has_edge_key(1, 2, k1))

    assert_equal(g.number_of_edges(), 2)

    var es = g.edges()
    assert_equal(len(es), 2)

    var w0 = g.get_edge_attr(1, 2, k0, "weight")
    assert_equal(w0[Float64], 3.0)

    var new_w = AttrValue(7.0)
    g.set_edge_attr(1, 2, k1, "weight", new_w)
    var w1 = g.get_edge_attr(1, 2, k1, "weight")
    assert_equal(w1[Float64], 7.0)

    assert_equal(g.degree(1), 2)
    assert_equal(g.degree(2), 2)

    g.remove_edge(1, 2, k0)
    assert_false(g.has_edge_key(1, 2, k0))
    assert_equal(g.number_of_edges(), 1)
