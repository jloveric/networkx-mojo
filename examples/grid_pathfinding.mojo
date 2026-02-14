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

This program renders a matplotlib figure saved as
  images/grid_paths.png
via Mojo's Python interop.

Run:
    mojo run -I . examples/grid_pathfinding.mojo
"""

from networkx import Graph
from math import sqrt
from python import Python, PythonObject


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

    try:
        var matplotlib: PythonObject = Python.import_module("matplotlib")
        matplotlib.use("Agg")
        var plt: PythonObject = Python.import_module("matplotlib.pyplot")
        var mpatches: PythonObject = Python.import_module("matplotlib.patches")
        var builtins: PythonObject = Python.import_module("builtins")
        var os: PythonObject = Python.import_module("os")
        os.makedirs("images", exist_ok=True)

        # Build a per-cell weight for the heatmap (max outgoing edge weight).
        var display = Python.list()
        r = 0
        while r < ROWS:
            var row = Python.list()
            var c = 0
            while c < COLS:
                var nid = r * COLS + c
                var w: Float64 = 1.0
                if g.has_node(nid):
                    for e in g._adj[nid].items():
                        if e.value > w:
                            w = e.value
                row.append(w)
                c += 1
            display.append(row)
            r += 1

        var max_w = sqrt(2.0)

        var path_keys = List[String]()
        var path_titles = List[String]()
        var path_colors = List[String]()
        var path_costs = List[Float64]()
        var path_lists = List[List[Int]]()

        path_keys.append("PATH_BFS")
        path_titles.append("BFS (min steps)")
        path_colors.append("#2196F3")
        path_costs.append(cost_bfs)
        path_lists.append(path_bfs.copy())

        path_keys.append("PATH_BIDIR_BFS")
        path_titles.append("Bidirectional BFS (min steps)")
        path_colors.append("#4CAF50")
        path_costs.append(cost_bidir)
        path_lists.append(path_bidir_bfs.copy())

        path_keys.append("PATH_DIJKSTRA")
        path_titles.append("Dijkstra (min total weight)")
        path_colors.append("#FF9800")
        path_costs.append(cost_dij)
        path_lists.append(path_dijkstra.copy())

        path_keys.append("PATH_BIDIR_DIJKSTRA")
        path_titles.append("Bidir-Dijkstra (min weight)")
        path_colors.append("#E91E63")
        path_costs.append(cost_bdij)
        path_lists.append(path_bidir_dij.copy())

        path_keys.append("PATH_ASTAR")
        path_titles.append("A* Euclidean (min weight)")
        path_colors.append("#9C27B0")
        path_costs.append(cost_astar)
        path_lists.append(path_astar.copy())

        path_keys.append("PATH_ASTAR_STEPS")
        path_titles.append("A* Chebyshev (min steps)")
        path_colors.append("#3F51B5")
        path_costs.append(cost_astar_steps)
        path_lists.append(path_astar_steps.copy())

        var n_paths = len(path_keys)
        var figsize_list = Python.list()
        figsize_list.append(4.8 * Float64(n_paths))
        figsize_list.append(6.0)
        var figsize: PythonObject = builtins.tuple(figsize_list)
        var fig: PythonObject = plt.figure(figsize=figsize)

        var i = 0
        while i < n_paths:
            var ax: PythonObject = fig.add_subplot(1, n_paths, i + 1)
            var title = path_titles[i]
            var colour = path_colors[i]
            var cost = path_costs[i]
            var node_list = path_lists[i].copy()

            var extent_list = Python.list()
            extent_list.append(-0.5)
            extent_list.append(Float64(COLS) - 0.5)
            extent_list.append(Float64(ROWS) - 0.5)
            extent_list.append(-0.5)
            var extent: PythonObject = builtins.tuple(extent_list)

            ax.imshow(
                display,
                origin="upper",
                extent=extent,
                cmap="YlOrRd",
                vmin=1.0,
                vmax=max_w,
                alpha=0.7,
            )

            for obs in obstacles:
                var orow = obs // COLS
                var ocol = obs % COLS
                var xy_list = Python.list()
                xy_list.append(Float64(ocol) - 0.5)
                xy_list.append(Float64(orow) - 0.5)
                var xy: PythonObject = builtins.tuple(xy_list)
                var rect = mpatches.Rectangle(
                    xy,
                    1.0,
                    1.0,
                    facecolor="#37474F",
                    edgecolor="#263238",
                    linewidth=0.6,
                )
                ax.add_patch(rect)

            r = 0
            while r <= ROWS:
                ax.axhline(Float64(r) - 0.5, color="#BDBDBD", linewidth=0.3)
                r += 1
            var cc = 0
            while cc <= COLS:
                ax.axvline(Float64(cc) - 0.5, color="#BDBDBD", linewidth=0.3)
                cc += 1

            var pc = Python.list()
            var pr = Python.list()
            for n in node_list:
                pr.append(n // COLS)
                pc.append(n % COLS)

            # White outline then coloured path
            ax.plot(
                pc,
                pr,
                color="white",
                linewidth=4.5,
                zorder=2,
                alpha=0.5,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            ax.plot(
                pc,
                pr,
                color=colour,
                linewidth=2.8,
                zorder=3,
                alpha=0.9,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

            var sr = source // COLS
            var sc = source % COLS
            var tr = target // COLS
            var tc = target % COLS

            ax.plot(
                sc,
                sr,
                marker="o",
                markersize=11,
                color="#00C853",
                markeredgecolor="black",
                markeredgewidth=1.3,
                zorder=5,
            )
            ax.plot(
                tc,
                tr,
                marker="*",
                markersize=15,
                color="#FF1744",
                markeredgecolor="black",
                markeredgewidth=1.0,
                zorder=5,
            )

            ax.set_title(
                title + "\n" + String(len(node_list) - 1) + " steps · weight " + String(cost),
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlim(-0.5, Float64(COLS) - 0.5)
            ax.set_ylim(Float64(ROWS) - 0.5, -0.5)
            ax.set_xticks(Python.list())
            ax.set_yticks(Python.list())
            ax.set_aspect("equal")

            i += 1

        fig.suptitle("Grid Pathfinding — networkx-mojo", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig("images/grid_paths.png", dpi=150, bbox_inches="tight", facecolor="white")
        print("Figure saved to images/grid_paths.png")
    except:
        print("Skipping plot: Python module 'matplotlib' is not installed.")
        print("To enable plotting, install optional deps and re-run:")
        print("    uv pip install -e '.[bench]'")

