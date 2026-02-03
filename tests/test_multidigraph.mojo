from networkx import MultiDiGraph
from collections import Dict
from testing import assert_equal, assert_true, assert_false
from utils import Variant


comptime AttrValue = Variant[Int, Float64, Bool, String]


fn main() raises:
    var g = MultiDiGraph[Int]()
    var k0 = g.add_edge(1, 2, 3.0)
    var k1 = g.add_edge(1, 2, 5.0)

    assert_true(g.is_multigraph())

    assert_true(g.has_edge(1, 2))
    assert_true(g.has_edge_key(1, 2, k0))
    assert_true(g.has_edge_key(1, 2, k1))
    assert_false(g.has_edge(2, 1))

    assert_equal(g.number_of_edges(), 2)
    assert_equal(g.out_degree(1), 2)
    assert_equal(g.in_degree(2), 2)

    ref succ_map = g.succ()
    assert_true(1 in succ_map)
    assert_true(2 in succ_map[1])
    assert_equal(len(succ_map[1][2]), 2)

    ref adj_map = g.adj()
    assert_true(1 in adj_map)
    assert_true(2 in adj_map[1])

    var succ1 = g[1]
    assert_true(2 in succ1)

    assert_equal(g.degree(1), 2)
    assert_equal(g.degree(2), 2)

    var succ = g.successors(1)
    assert_equal(len(succ), 1)
    assert_equal(succ[0], 2)

    var pred = g.predecessors(2)
    assert_equal(len(pred), 1)
    assert_equal(pred[0], 1)

    var n1 = g.neighbors(1)
    assert_equal(len(n1), 1)
    assert_equal(n1[0], 2)

    var w0 = g.get_edge_attr(1, 2, k0, "weight")
    assert_equal(w0[Float64], 3.0)

    var new_w = AttrValue(7.0)
    g.set_edge_attr(1, 2, k1, "weight", new_w)
    var w1 = g.get_edge_attr(1, 2, k1, "weight")
    assert_equal(w1[Float64], 7.0)

    g.remove_edge(1, 2, k0)
    assert_false(g.has_edge_key(1, 2, k0))
    assert_equal(g.number_of_edges(), 1)

    g.remove_edges_from([(1, 2, k0), (1, 2, 99)])
    assert_equal(g.number_of_edges(), 1)

    var g2 = MultiDiGraph[Int]()
    _ = g2.add_edge(1, 2)
    _ = g2.add_edge(3, 1)
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

    var g3 = MultiDiGraph[Int]()
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

    var gt = MultiDiGraph[Int]()
    _ = gt.add_edge(1, 2)
    _ = gt.add_edge(1, 2)
    _ = gt.add_edge(2, 3)
    _ = gt.add_edge(2, 4)

    var be = gt.bfs_edges(1)
    assert_equal(len(be), 3)
    var seen_child = Dict[Int, Bool]()
    for e in be:
        seen_child[e[1]] = True
    assert_true(2 in seen_child)
    assert_true(3 in seen_child)
    assert_true(4 in seen_child)

    var bt = gt.bfs_tree(1)
    assert_equal(bt.number_of_nodes(), 4)
    assert_equal(bt.number_of_edges(), 3)

    var de = gt.dfs_edges(1)
    assert_equal(len(de), 3)
    var seen_child2 = Dict[Int, Bool]()
    for e in de:
        seen_child2[e[1]] = True
    assert_true(2 in seen_child2)
    assert_true(3 in seen_child2)
    assert_true(4 in seen_child2)

    var dt = gt.dfs_tree(1)
    assert_equal(dt.number_of_nodes(), 4)
    assert_equal(dt.number_of_edges(), 3)

    var bp = gt.bfs_predecessors(1)
    assert_equal(len(bp), 3)
    var parents = Dict[Int, Int]()
    for kv in bp:
        parents[kv[0]] = kv[1]
    assert_true(2 in parents)
    assert_true(3 in parents)
    assert_true(4 in parents)

    var bs = gt.bfs_successors(1)
    var succ_seen = Dict[Int, Bool]()
    for kv in bs:
        for v in kv[1]:
            succ_seen[v] = True
    assert_true(2 in succ_seen)
    assert_true(3 in succ_seen)
    assert_true(4 in succ_seen)

    var dp = gt.dfs_predecessors(1)
    assert_equal(len(dp), 3)
    var ds = gt.dfs_successors(1)
    var succ_seen2 = Dict[Int, Bool]()
    for kv in ds:
        for v in kv[1]:
            succ_seen2[v] = True
    assert_true(2 in succ_seen2)
    assert_true(3 in succ_seen2)
    assert_true(4 in succ_seen2)

    var layers = gt.bfs_layers(1)
    assert_equal(len(layers), 3)
    assert_equal(layers[0][0], 1)
