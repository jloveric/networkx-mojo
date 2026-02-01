from networkx import Graph
from testing import assert_equal, assert_true, assert_false


fn _noop_node(x: Int):
    _ = x


fn _noop_edge(u: Int, v: Int):
    _ = u
    _ = v


fn main() raises:
    var g = Graph[Int]()
    assert_equal(g.number_of_nodes(), 0)
    assert_equal(g.number_of_edges(), 0)
    assert_equal(g.order(), 0)
    assert_equal(g.size(), 0)
    assert_false(g.is_directed())

    g.add_node(1)
    assert_true(g.has_node(1))
    assert_equal(g.number_of_nodes(), 1)

    g.add_edge(1, 2)
    assert_true(g.has_edge(1, 2))
    assert_true(g.has_edge(2, 1))
    assert_equal(g.number_of_nodes(), 2)
    assert_equal(g.number_of_edges(), 1)

    assert_equal(g.for_each_node[_noop_node](), 2)
    assert_equal(g.for_each_neighbor[_noop_node](1), 1)
    assert_equal(g.for_each_edge[_noop_edge](), 1)

    assert_false(g.has_edge(1, 3))

    var edges = g.edges()
    assert_equal(len(edges), 1)
    assert_true((edges[0] == (1, 2)) or (edges[0] == (2, 1)))

    var n = 0
    for _ in g.nodes():
        n += 1
    assert_equal(n, 2)

    var n2 = 0
    for _ in g.adj_view().keys():
        n2 += 1
    assert_equal(n2, 2)

    var deg1 = 0
    for _ in g.neighbors(1):
        deg1 += 1
    assert_equal(deg1, 1)

    var deg1b = 0
    ref nbrs = g.adj_view()[1]
    for _ in nbrs.keys():
        deg1b += 1
    assert_equal(deg1b, 1)

    g.remove_edge(1, 2)
    assert_false(g.has_edge(1, 2))
    assert_equal(g.number_of_edges(), 0)

    g.add_edge(1, 2)
    g.add_edge(2, 3)
    assert_equal(g.number_of_edges(), 2)
    g.remove_node(2)
    assert_false(g.has_node(2))
    assert_false(g.has_edge(1, 2))
    assert_false(g.has_edge(2, 3))
    assert_equal(g.number_of_edges(), 0)

    var g2 = Graph[Int]()
    g2.add_nodes_from([1, 2, 3])
    assert_equal(g2.number_of_nodes(), 3)
    g2.add_edges_from([(1, 2), (2, 3)])
    assert_equal(g2.number_of_edges(), 2)
    assert_equal(g2.degree(2), 2)

    g2.add_edge(1, 1)
    assert_equal(g2.number_of_edges(), 3)
    assert_equal(g2.degree(1), 3)

    assert_equal(g2.for_each_edge[_noop_edge](), g2.number_of_edges())

    g2.clear()
    assert_equal(g2.number_of_nodes(), 0)
    assert_equal(g2.number_of_edges(), 0)
