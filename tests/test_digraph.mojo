from networkx import DiGraph
from testing import assert_equal, assert_true, assert_false


fn _noop_node(x: Int):
    _ = x


fn _noop_edge(u: Int, v: Int):
    _ = u
    _ = v


fn main() raises:
    var g = DiGraph[Int]()
    assert_equal(g.number_of_nodes(), 0)
    assert_equal(g.number_of_edges(), 0)
    assert_equal(g.order(), 0)
    assert_equal(g.size(), 0)
    assert_true(g.is_directed())

    g.add_edge(1, 2)
    assert_true(g.has_node(1))
    assert_true(g.has_node(2))
    assert_true(g.has_edge(1, 2))
    assert_false(g.has_edge(2, 1))
    assert_equal(g.number_of_edges(), 1)

    assert_equal(g.for_each_node[_noop_node](), 2)
    assert_equal(g.for_each_successor[_noop_node](1), 1)
    assert_equal(g.for_each_predecessor[_noop_node](2), 1)
    assert_equal(g.for_each_edge[_noop_edge](), 1)

    var s = g.successors(1)
    assert_equal(len(s), 1)
    assert_equal(s[0], 2)

    var n = g.neighbors(1)
    assert_equal(len(n), 1)
    assert_equal(n[0], 2)

    var a = g.adj(1)
    assert_equal(len(a), 1)
    assert_equal(a[0], 2)

    var p = g.predecessors(2)
    assert_equal(len(p), 1)
    assert_equal(p[0], 1)

    g.add_edge(2, 3)
    assert_equal(g.number_of_nodes(), 3)
    assert_equal(g.number_of_edges(), 2)

    g.remove_edge(1, 2)
    assert_false(g.has_edge(1, 2))
    assert_equal(g.number_of_edges(), 1)

    g.add_edge(1, 2)
    g.add_edge(3, 2)
    assert_true(g.has_edge(3, 2))
    g.remove_node(2)
    assert_false(g.has_node(2))
    assert_false(g.has_edge(1, 2))
    assert_false(g.has_edge(3, 2))
    assert_false(g.has_edge(2, 3))

    var g2 = DiGraph[Int]()
    g2.add_nodes_from([1, 2, 3])
    g2.add_edges_from([(1, 2), (1, 3), (2, 3)])
    assert_equal(g2.number_of_edges(), 3)
    assert_equal(g2.out_degree(1), 2)
    assert_equal(g2.in_degree(3), 2)
    assert_equal(g2.degree(3), 2)

    g2.clear()
    assert_equal(g2.number_of_nodes(), 0)
    assert_equal(g2.number_of_edges(), 0)
