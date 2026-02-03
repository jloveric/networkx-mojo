from networkx import MultiGraph
from testing import assert_equal, assert_true, assert_false
from utils import Variant


comptime AttrValue = Variant[Int, Float64, Bool, String]


fn main() raises:
    var g = MultiGraph[Int]()
    var k0 = g.add_edge(1, 2, 3.0)
    var k1 = g.add_edge(1, 2, 5.0)

    assert_true(g.is_multigraph())

    assert_true(g.has_node(1))
    assert_true(g.has_node(2))

    assert_true(g.has_edge(1, 2))
    assert_true(g.has_edge_key(1, 2, k0))
    assert_true(g.has_edge_key(1, 2, k1))

    assert_equal(g.number_of_edges(), 2)

    var es = g.edges()
    assert_equal(len(es), 2)

    var n1 = g.neighbors(1)
    assert_equal(len(n1), 1)
    assert_equal(n1[0], 2)

    ref adj = g.adj_view()
    assert_true(1 in adj)
    assert_true(2 in adj)

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

    g.remove_edges_from([(1, 2, k0), (1, 2, 99)])
    assert_equal(g.number_of_edges(), 1)

    var g2 = MultiGraph[Int]()
    _ = g2.add_edge(1, 2)
    _ = g2.add_edge(1, 3)
    assert_equal(g2.number_of_edges(), 2)
    g2.remove_node(1)
    assert_false(g2.has_node(1))
    assert_equal(g2.number_of_edges(), 0)

    g2.remove_nodes_from([1, 99])
    assert_equal(g2.number_of_nodes(), 2)

    _ = g2.add_edge(4, 5)
    assert_equal(g2.number_of_edges(), 1)
    g2.clear()
    assert_equal(g2.number_of_nodes(), 0)
    assert_equal(g2.number_of_edges(), 0)

    var g3 = MultiGraph[Int]()
    _ = g3.add_edge(1, 2, 2.0)
    _ = g3.add_edge(2, 3, 3.0)
    g3.set_graph_attr("name", AttrValue("g3"))
    g3.set_node_attr(2, "color", AttrValue("blue"))
    var cap = AttrValue(7)
    g3.set_edge_attr(1, 2, 0, "capacity", cap)

    var g3c = g3.copy()
    assert_equal(g3c.number_of_nodes(), 3)
    assert_equal(g3c.number_of_edges(), 2)
    assert_equal(g3c.get_graph_attr("name")[String], "g3")
    assert_equal(g3c.get_node_attr(2, "color")[String], "blue")
    assert_equal(g3c.get_edge_attr(1, 2, 0, "capacity")[Int], 7)

    var sg = g3.subgraph([2, 3])
    assert_equal(sg.number_of_nodes(), 2)
    assert_equal(sg.number_of_edges(), 1)
    assert_equal(sg.get_graph_attr("name")[String], "g3")
    assert_equal(sg.get_node_attr(2, "color")[String], "blue")
