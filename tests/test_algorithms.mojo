from networkx import Graph, DiGraph
from testing import assert_equal, assert_true


fn w(u: Int, v: Int) -> Float64:
    if u == 1 and v == 2:
        return 5.0
    if u == 1 and v == 3:
        return 1.0
    if u == 3 and v == 2:
        return 1.0
    return 1.0


fn h(u: Int, target: Int) -> Float64:
    return 0.0


fn main() raises:
    var g = Graph[Int]()
    g.add_edges_from([(1, 2), (2, 3), (1, 3)])

    var p1 = g.shortest_path(1, 3)
    assert_equal(p1[0], 1)
    assert_equal(p1[len(p1) - 1], 3)

    var p2 = g.dijkstra_path_weighted[w](1, 2)
    assert_equal(len(p2), 3)
    assert_equal(p2[0], 1)
    assert_equal(p2[1], 3)
    assert_equal(p2[2], 2)

    var p3 = g.astar_path_weighted[w, h](1, 2)
    assert_equal(len(p3), 3)
    assert_equal(p3[0], 1)
    assert_equal(p3[1], 3)
    assert_equal(p3[2], 2)

    var gw = Graph[Int]()
    gw.add_edge(1, 2, 5.0)
    gw.add_edge(1, 3, 1.0)
    gw.add_edge(3, 2, 1.0)
    var dpw = gw.dijkstra_path(1, 2)
    assert_equal(len(dpw), 3)
    assert_equal(dpw[0], 1)
    assert_equal(dpw[1], 3)
    assert_equal(dpw[2], 2)
    var apw = gw.astar_path(1, 2)
    assert_equal(len(apw), 3)
    assert_equal(apw[0], 1)
    assert_equal(apw[1], 3)
    assert_equal(apw[2], 2)

    var dg = DiGraph[Int]()
    dg.add_edges_from([(1, 2), (1, 3), (3, 2)])

    var dp = dg.shortest_path(1, 2)
    assert_equal(dp[0], 1)
    assert_equal(dp[len(dp) - 1], 2)

    var d2 = dg.dijkstra_path_weighted[w](1, 2)
    assert_equal(len(d2), 3)
    assert_equal(d2[0], 1)
    assert_equal(d2[1], 3)
    assert_equal(d2[2], 2)

    var a2 = dg.astar_path_weighted[w, h](1, 2)
    assert_equal(len(a2), 3)
    assert_equal(a2[0], 1)
    assert_equal(a2[1], 3)
    assert_equal(a2[2], 2)

    var dgw = DiGraph[Int]()
    dgw.add_edge(1, 2, 5.0)
    dgw.add_edge(1, 3, 1.0)
    dgw.add_edge(3, 2, 1.0)
    var ddpw = dgw.dijkstra_path(1, 2)
    assert_equal(len(ddpw), 3)
    assert_equal(ddpw[0], 1)
    assert_equal(ddpw[1], 3)
    assert_equal(ddpw[2], 2)
    var dapw = dgw.astar_path(1, 2)
    assert_equal(len(dapw), 3)
    assert_equal(dapw[0], 1)
    assert_equal(dapw[1], 3)
    assert_equal(dapw[2], 2)

    var caught = False
    try:
        _ = g.shortest_path(1, 999)
    except:
        caught = True
    assert_true(caught)
