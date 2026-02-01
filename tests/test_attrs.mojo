from networkx import Graph, DiGraph
from testing import assert_equal, assert_true, assert_false
from utils import Variant


comptime AttrValue = Variant[Int, Float64, Bool, String]


fn main() raises:
    var g = Graph[Int]()
    g.add_edge(1, 2, 2.5)

    g.set_graph_attr("answer", AttrValue(42))
    var ga = g.get_graph_attr("answer")
    assert_true(ga.isa[Int]())
    assert_equal(ga[Int], 42)

    g.set_node_attr(1, "active", AttrValue(True))
    var na = g.get_node_attr(1, "active")
    assert_true(na.isa[Bool]())
    assert_true(na[Bool])

    var w1 = g.get_edge_attr(1, 2, "weight")
    assert_true(w1.isa[Float64]())
    assert_equal(w1[Float64], 2.5)

    var cap = AttrValue(7)
    g.set_edge_attr(1, 2, "capacity", cap)
    var c1 = g.get_edge_attr(1, 2, "capacity")
    assert_true(c1.isa[Int]())
    assert_equal(c1[Int], 7)

    var c2 = g.get_edge_attr(2, 1, "capacity")
    assert_true(c2.isa[Int]())
    assert_equal(c2[Int], 7)

    var new_w = AttrValue(9.0)
    g.set_edge_attr(1, 2, "weight", new_w)
    var w2 = g.get_edge_attr(2, 1, "weight")
    assert_true(w2.isa[Float64]())
    assert_equal(w2[Float64], 9.0)

    var dg = DiGraph[Int]()
    dg.add_edge(1, 2, 1.0)

    dg.set_node_attr(1, "label", AttrValue("src"))
    var lbl = dg.get_node_attr(1, "label")
    assert_true(lbl.isa[String]())
    assert_equal(lbl[String], "src")

    var dcap = AttrValue(3)
    dg.set_edge_attr(1, 2, "capacity", dcap)
    var dc1 = dg.get_edge_attr(1, 2, "capacity")
    assert_true(dc1.isa[Int]())
    assert_equal(dc1[Int], 3)

    var caught = False
    try:
        _ = dg.get_edge_attr(2, 1, "capacity")
    except:
        caught = True
    assert_true(caught)
