from networkx import Graph, DiGraph
from collections import Dict
from testing import assert_equal, assert_true, assert_false


fn main() raises:
    var g = Graph[Int]()
    g.add_edges_from([(1, 2), (2, 3), (4, 5)])

    var comps = g.connected_components()
    assert_equal(len(comps), 2)
    assert_equal(g.number_connected_components(), 2)
    assert_false(g.is_connected())

    var total = 0
    for c in comps:
        total += len(c)
    assert_equal(total, 5)

    var g_conn = Graph[Int]()
    g_conn.add_edges_from([(1, 2), (2, 3)])
    assert_equal(g_conn.number_connected_components(), 1)
    assert_true(g_conn.is_connected())

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

    assert_equal(mst.shortest_path_length(1, 2), 2)
    var sp_len = mst.single_source_shortest_path_length(1)
    assert_equal(sp_len[1], 0)
    assert_equal(sp_len[2], 2)

    var sp = mst.single_source_shortest_path(1)
    assert_equal(sp[1][0], 1)
    assert_equal(sp[2][0], 1)
    assert_equal(sp[2][len(sp[2]) - 1], 2)

    var dj_len = mst.dijkstra_path_length(1, 2)
    assert_equal(dj_len, 3.0)

    var ss_dj_len = mst.single_source_dijkstra_path_length(1)
    assert_equal(ss_dj_len[1], 0.0)
    assert_equal(ss_dj_len[2], 3.0)

    var ss_dj = mst.single_source_dijkstra_path(1)
    assert_equal(ss_dj[2][0], 1)
    assert_equal(ss_dj[2][len(ss_dj[2]) - 1], 2)

    var g_bfs = Graph[Int]()
    g_bfs.add_edges_from([(1, 2), (2, 3), (2, 4)])
    var be = g_bfs.bfs_edges(1)
    assert_equal(len(be), 3)
    var seen_child = Dict[Int, Bool]()
    for e in be:
        seen_child[e[1]] = True
    assert_true(2 in seen_child)
    assert_true(3 in seen_child)
    assert_true(4 in seen_child)

    var bt = g_bfs.bfs_tree(1)
    assert_equal(bt.number_of_nodes(), 4)
    assert_equal(bt.number_of_edges(), 3)

    var de = g_bfs.dfs_edges(1)
    assert_equal(len(de), 3)
    var seen_child2 = Dict[Int, Bool]()
    for e in de:
        seen_child2[e[1]] = True
    assert_true(2 in seen_child2)
    assert_true(3 in seen_child2)
    assert_true(4 in seen_child2)

    var dt = g_bfs.dfs_tree(1)
    assert_equal(dt.number_of_nodes(), 4)
    assert_equal(dt.number_of_edges(), 3)

    var bp = g_bfs.bfs_predecessors(1)
    assert_equal(len(bp), 3)
    var parents = Dict[Int, Int]()
    for kv in bp:
        parents[kv[0]] = kv[1]
    assert_true(2 in parents)
    assert_true(3 in parents)
    assert_true(4 in parents)

    var bs = g_bfs.bfs_successors(1)
    var succ_seen = Dict[Int, Bool]()
    for kv in bs:
        for v in kv[1]:
            succ_seen[v] = True
    assert_true(2 in succ_seen)
    assert_true(3 in succ_seen)
    assert_true(4 in succ_seen)

    var dp = g_bfs.dfs_predecessors(1)
    assert_equal(len(dp), 3)
    var ds = g_bfs.dfs_successors(1)
    var succ_seen2 = Dict[Int, Bool]()
    for kv in ds:
        for v in kv[1]:
            succ_seen2[v] = True
    assert_true(2 in succ_seen2)
    assert_true(3 in succ_seen2)
    assert_true(4 in succ_seen2)

    var layers = g_bfs.bfs_layers(1)
    assert_equal(len(layers), 3)
    assert_equal(layers[0][0], 1)

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

    assert_true(dg.is_dag())
    var desc = dg.descendants(5)
    var seen_desc = Dict[Int, Bool]()
    for n in desc:
        seen_desc[n] = True
    assert_true(6 in seen_desc)
    assert_true(7 in seen_desc)
    assert_true(8 in seen_desc)
    assert_false(5 in seen_desc)

    var anc = dg.ancestors(8)
    var seen_anc = Dict[Int, Bool]()
    for n in anc:
        seen_anc[n] = True
    assert_true(5 in seen_anc)
    assert_true(6 in seen_anc)
    assert_true(7 in seen_anc)
    assert_false(8 in seen_anc)

    assert_equal(dg.shortest_path_length(5, 8), 2)
    var dg_sp_len = dg.single_source_shortest_path_length(5)
    assert_equal(dg_sp_len[5], 0)
    assert_equal(dg_sp_len[8], 2)

    var dg_sp = dg.single_source_shortest_path(5)
    assert_equal(dg_sp[5][0], 5)
    assert_equal(dg_sp[8][0], 5)
    assert_equal(dg_sp[8][len(dg_sp[8]) - 1], 8)

    var cyc = DiGraph[Int]()
    cyc.add_edges_from([(1, 2), (2, 1)])
    var caught2 = False
    try:
        _ = cyc.topological_sort()
    except:
        caught2 = True
    assert_true(caught2)

    assert_false(cyc.is_dag())

    var dg2 = DiGraph[Int]()
    dg2.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    var dbe = dg2.bfs_edges(1)
    assert_equal(len(dbe), 3)
    var dseen = Dict[Int, Bool]()
    for e in dbe:
        dseen[e[1]] = True
    assert_true(2 in dseen)
    assert_true(3 in dseen)
    assert_true(4 in dseen)

    var dbt = dg2.bfs_tree(1)
    assert_equal(dbt.number_of_nodes(), 4)
    assert_equal(dbt.number_of_edges(), 3)

    var dde = dg2.dfs_edges(1)
    assert_equal(len(dde), 3)
    var dseen2 = Dict[Int, Bool]()
    for e in dde:
        dseen2[e[1]] = True
    assert_true(2 in dseen2)
    assert_true(3 in dseen2)
    assert_true(4 in dseen2)

    var ddt = dg2.dfs_tree(1)
    assert_equal(ddt.number_of_nodes(), 4)
    assert_equal(ddt.number_of_edges(), 3)

    var dbp = dg2.bfs_predecessors(1)
    assert_equal(len(dbp), 3)
    var dparents = Dict[Int, Int]()
    for kv in dbp:
        dparents[kv[0]] = kv[1]
    assert_true(2 in dparents)
    assert_true(3 in dparents)
    assert_true(4 in dparents)

    var dbl = dg2.bfs_layers(1)
    assert_equal(len(dbl), 3)
    assert_equal(dbl[0][0], 1)
