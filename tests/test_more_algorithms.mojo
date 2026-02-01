from networkx import Graph, DiGraph
from collections import Dict
from testing import assert_equal, assert_true


fn main() raises:
    var g = Graph[Int]()
    g.add_edges_from([(1, 2), (2, 3), (4, 5)])

    var comps = g.connected_components()
    assert_equal(len(comps), 2)

    var total = 0
    for c in comps:
        total += len(c)
    assert_equal(total, 5)

    var gw = Graph[Int]()
    gw.add_edge(1, 2, 10.0)
    gw.add_edge(2, 3, 1.0)
    gw.add_edge(1, 3, 2.0)
    var mst = gw.minimum_spanning_tree()
    assert_equal(mst.number_of_edges(), 2)

    var p = mst.dijkstra_path(1, 2)
    assert_equal(len(p), 3)
    assert_equal(p[0], 1)
    assert_equal(p[1], 3)
    assert_equal(p[2], 2)

    var dg = DiGraph[Int]()
    dg.add_edges_from([(5, 6), (5, 7), (6, 8), (7, 8)])
    var order = dg.topological_sort()
    assert_equal(len(order), 4)

    var pos = Dict[Int, Int]()
    var i = 0
    for n in order:
        pos[n] = i
        i += 1

    assert_true(pos[5] < pos[6])
    assert_true(pos[5] < pos[7])
    assert_true(pos[6] < pos[8])
    assert_true(pos[7] < pos[8])

    var cyc = DiGraph[Int]()
    cyc.add_edges_from([(1, 2), (2, 1)])
    var caught2 = False
    try:
        _ = cyc.topological_sort()
    except:
        caught2 = True
    assert_true(caught2)
