"""
Grid Pathfinding Example
========================
Builds a 15×20 grid graph with obstacles and edge weights equal to
segment length (orthogonal = 1, diagonal = sqrt(2)),
then finds paths from the top-left corner (0,0) to the bottom-right corner
using several search algorithms provided by the networkx-mojo Graph API:

  1. BFS shortest path        (fewest hops – ignores weights)
  2. Bidirectional BFS         (fewest hops – ignores weights)
  3. Dijkstra                  (minimum total weight)
  4. Bidirectional Dijkstra    (minimum total weight)
  5. A* search                 (Euclidean heuristic + stored edge weights)

Because BFS minimises *hops* while Dijkstra minimises *total weight*,
the two families of algorithms will generally find **different** routes
through the grid.  A* with a Euclidean heuristic explores fewer nodes
than Dijkstra while finding the same optimal weighted path.

The grid, obstacles, weights, and every path are written to
  examples/grid_paths.txt
which the companion script  examples/visualize_grid.py  reads to
produce a matplotlib figure saved as  examples/grid_paths.png.

Run:
    mojo run -I . examples/grid_pathfinding.mojo
    python examples/visualize_grid.py
"""

from networkx import Graph
from math import sqrt


comptime ROWS = 15
comptime COLS = 20


fn _segment_length(u: Int, v: Int) -> Float64:
    var ru = u // COLS
    var cu = u % COLS
    var rv = v // COLS
    var cv = v % COLS

    var dr = ru - rv
    if dr < 0:
        dr = -dr
    var dc = cu - cv
    if dc < 0:
        dc = -dc
    if dr == 1 and dc == 1:
        return sqrt(2.0)
    return 1.0


fn _euclidean_heuristic(u: Int, target: Int) -> Float64:
    var ru = u // COLS
    var cu = u % COLS
    var rt = target // COLS
    var ct = target % COLS
    var dr = Float64(ru - rt)
    var dc = Float64(cu - ct)
    return sqrt(dr * dr + dc * dc)


fn _unit_weight(u: Int, v: Int) -> Float64:
    _ = u
    _ = v
    return 1.0


fn _chebyshev_heuristic(u: Int, target: Int) -> Float64:
    var ru = u // COLS
    var cu = u % COLS
    var rt = target // COLS
    var ct = target % COLS
    var dr = ru - rt
    if dr < 0:
        dr = -dr
    var dc = cu - ct
    if dc < 0:
        dc = -dc
    if dr > dc:
        return Float64(dr)
    return Float64(dc)


fn main() raises:
    var g = Graph[Int]()

    # Add all nodes
    var r = 0
    while r < ROWS:
        var c = 0
        while c < COLS:
            g.add_node(r * COLS + c)
            c += 1
        r += 1

    # Add edges once (to "forward" neighbors), including diagonals
    r = 0
    while r < ROWS:
        var c = 0
        while c < COLS:
            var u = r * COLS + c
            if c + 1 < COLS:
                g.add_edge(u, r * COLS + (c + 1), 1.0)
            if r + 1 < ROWS:
                g.add_edge(u, (r + 1) * COLS + c, 1.0)

            # Diagonal moves
            if (r + 1 < ROWS) and (c + 1 < COLS):
                g.add_edge(u, (r + 1) * COLS + (c + 1), sqrt(2.0))
            if (r + 1 < ROWS) and (c - 1 >= 0):
                g.add_edge(u, (r + 1) * COLS + (c - 1), sqrt(2.0))

            c += 1
        r += 1

    var obstacles = List[Int]()

    var wc = 10
    while wc <= 14:
        obstacles.append(0 * COLS + wc)
        wc += 1

    var wr = 3
    while wr <= 7:
        obstacles.append(wr * COLS + (COLS - 1))
        wr += 1

    wr = 2
    while wr <= 8:
        obstacles.append(wr * COLS + 5)
        wr += 1

    wc = 8
    while wc <= 13:
        obstacles.append(4 * COLS + wc)
        wc += 1

    wr = 6
    while wr <= 12:
        obstacles.append(wr * COLS + 10)
        wr += 1

    wc = 3
    while wc <= 8:
        obstacles.append(10 * COLS + wc)
        wc += 1

    wc = 12
    while wc <= 17:
        obstacles.append(7 * COLS + wc)
        wc += 1

    for obs in obstacles:
        if g.has_node(obs):
            g.remove_node(obs)

    var source = 0
    var target = (ROWS - 1) * COLS + (COLS - 1)

    var path_bfs = g.shortest_path(source, target)
    var path_bidir_bfs = g.bidirectional_shortest_path(source, target)
    var path_dijkstra = g.dijkstra_path(source, target)
    var path_bidir_dij = g.bidirectional_dijkstra_path(source, target)
    var path_astar = g.astar_path_weighted[_segment_length, _euclidean_heuristic](source, target)
    var path_astar_steps = g.astar_path_weighted[_unit_weight, _chebyshev_heuristic](source, target)

    fn _path_weight(ref graph: Graph[Int], ref path: List[Int]) raises -> Float64:
        var total: Float64 = 0.0
        var i = 0
        while i < len(path) - 1:
            total += graph._adj[path[i]][path[i + 1]]
            i += 1
        return total

    var cost_bfs = _path_weight(g, path_bfs)
    var cost_bidir = _path_weight(g, path_bidir_bfs)
    var cost_dij = _path_weight(g, path_dijkstra)
    var cost_bdij = _path_weight(g, path_bidir_dij)
    var cost_astar = _path_weight(g, path_astar)
    var cost_astar_steps = _path_weight(g, path_astar_steps)

    var out = String()
    out += "GRID " + String(ROWS) + " " + String(COLS) + "\n"
    out += "OBSTACLES"
    for obs in obstacles:
        out += " " + String(obs)
    out += "\n"
    out += "WEIGHTS"
    for entry in g._adj.items():
        var u = entry.key
        for e in entry.value.items():
            var v = e.key
            if u < v:
                out += " " + String(u) + "," + String(v) + "," + String(e.value)
    out += "\n"
    out += "SOURCE " + String(source) + "\n"
    out += "TARGET " + String(target) + "\n"

    fn _write_path(mut buf: String, name: String, ref path: List[Int]):
        buf += name
        for i in range(len(path)):
            buf += " " + String(path[i])
        buf += "\n"

    _write_path(out, "PATH_BFS", path_bfs)
    _write_path(out, "PATH_BIDIR_BFS", path_bidir_bfs)
    _write_path(out, "PATH_DIJKSTRA", path_dijkstra)
    _write_path(out, "PATH_BIDIR_DIJKSTRA", path_bidir_dij)
    _write_path(out, "PATH_ASTAR", path_astar)
    _write_path(out, "PATH_ASTAR_STEPS", path_astar_steps)

    print("Grid:", ROWS, "x", COLS)
    print("Obstacles:", len(obstacles))
    print("Nodes in graph:", g.number_of_nodes())
    print("Edges in graph:", g.number_of_edges())
    print()
    print("BFS path length:              ", len(path_bfs) - 1, "steps, weight", cost_bfs)
    print("Bidirectional BFS path length:", len(path_bidir_bfs) - 1, "steps, weight", cost_bidir)
    print("Dijkstra path length:         ", len(path_dijkstra) - 1, "steps, weight", cost_dij)
    print("Bidir-Dijkstra path length:   ", len(path_bidir_dij) - 1, "steps, weight", cost_bdij)
    print("A* (Euclidean) path length:   ", len(path_astar) - 1, "steps, weight", cost_astar)
    print("A* (min steps) path length:   ", len(path_astar_steps) - 1, "steps, weight", cost_astar_steps)

    with open("examples/grid_paths.txt", "w") as f:
        f.write(out)

    print()
    print("Path data written to examples/grid_paths.txt")
    print("Run:  python examples/visualize_grid.py")
