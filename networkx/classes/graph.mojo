from builtin.value import ImplicitlyCopyable
from collections import Dict, List, Set
from collections.dict import KeyElement
from utils import Variant
from .._internal.heap import _HeapItem, _MinHeap

fn _unit_weight[N: KeyElement & ImplicitlyCopyable](u: N, v: N) -> Float64:
    return 1.0

fn _zero_heuristic[N: KeyElement & ImplicitlyCopyable](u: N, v: N) -> Float64:
    return 0.0

fn _reverse_in_place[N: KeyElement & ImplicitlyCopyable](mut path: List[N]):
    var i = 0
    var j = len(path) - 1
    while i < j:
        var tmp = path[i]
        path[i] = path[j]
        path[j] = tmp
        i += 1
        j -= 1


struct _WEdge[N: KeyElement & ImplicitlyCopyable](ImplicitlyCopyable):
    var u: Self.N
    var v: Self.N
    var w: Float64

    fn __init__(out self, u: Self.N, v: Self.N, w: Float64):
        self.u = u
        self.v = v
        self.w = w


fn _sort_wedges_in_place[N: KeyElement & ImplicitlyCopyable](mut edges: List[_WEdge[N]]):
    var i = 1
    while i < len(edges):
        var j = i
        while j > 0 and edges[j].w < edges[j - 1].w:
            var tmp = edges[j - 1]
            edges[j - 1] = edges[j]
            edges[j] = tmp
            j -= 1
        i += 1


struct _UnionFind[N: KeyElement & ImplicitlyCopyable]:
    var parent: Dict[Self.N, Self.N]
    var rank: Dict[Self.N, Int]

    fn __init__(out self):
        self.parent = Dict[Self.N, Self.N]()
        self.rank = Dict[Self.N, Int]()

    fn add(mut self, x: Self.N):
        if x in self.parent:
            return
        self.parent[x] = x
        self.rank[x] = 0

    fn find(mut self, x: Self.N) raises -> Self.N:
        var cur = x
        while self.parent[cur] != cur:
            cur = self.parent[cur]
        var root = cur

        cur = x
        while self.parent[cur] != cur:
            var p = self.parent[cur]
            self.parent[cur] = root
            cur = p
        return root

    fn union(mut self, a: Self.N, b: Self.N) raises -> Bool:
        var ra = self.find(a)
        var rb = self.find(b)
        if ra == rb:
            return False

        var rka = self.rank[ra]
        var rkb = self.rank[rb]
        if rka < rkb:
            self.parent[ra] = rb
        elif rka > rkb:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] = rka + 1
        return True

comptime AttrValue = Variant[Int, Float64, Bool, String]

struct Graph[N: KeyElement & ImplicitlyCopyable]:
    var _adj: Dict[Self.N, Dict[Self.N, Float64]]
    var _graph_attr: Dict[String, AttrValue]
    var _node_attr: Dict[Self.N, Dict[String, AttrValue]]
    var _edge_attr: Dict[Self.N, Dict[Self.N, Dict[String, AttrValue]]]

    fn __init__(out self):
        self._adj = Dict[Self.N, Dict[Self.N, Float64]]()
        self._graph_attr = Dict[String, AttrValue]()
        self._node_attr = Dict[Self.N, Dict[String, AttrValue]]()
        self._edge_attr = Dict[Self.N, Dict[Self.N, Dict[String, AttrValue]]]()

    fn _reconstruct_path(ref self, ref parents: Dict[Self.N, Self.N], source: Self.N, target: Self.N) raises -> List[Self.N]:
        var path = List[Self.N]()
        var cur = target
        path.append(cur)
        while cur != source:
            cur = parents[cur]
            path.append(cur)
        _reverse_in_place(path)
        return path^

    fn __len__(self) -> Int:
        return self.number_of_nodes()

    fn __contains__(self, node: Self.N) -> Bool:
        return self.has_node(node)

    fn __iter__(self) -> Dict[Self.N, Dict[Self.N, Float64]].IteratorType[iterable_mut=False, iterable_origin=origin_of(self._adj)]:
        return self._adj.keys()

    fn number_of_nodes(self) -> Int:
        return len(self._adj)

    fn order(self) -> Int:
        return self.number_of_nodes()

    fn number_of_edges(self) -> Int:
        var total_degree = 0
        var self_loops = 0
        for e in self._adj.items():
            total_degree += len(e.value)
            if e.key in e.value:
                self_loops += 1
        return (total_degree + self_loops) // 2

    fn size(self) -> Int:
        return self.number_of_edges()

    fn is_directed(self) -> Bool:
        return False

    fn is_multigraph(self) -> Bool:
        return False

    fn connected_components(ref self) raises -> List[List[Self.N]]:
        var seen = Set[Self.N]()
        var comps = List[List[Self.N]]()

        for node in self._adj.keys():
            if node in seen:
                continue

            var comp = List[Self.N]()
            var stack = List[Self.N]()
            stack.append(node)
            seen.add(node)

            while len(stack) > 0:
                var u = stack.pop()
                comp.append(u)
                for v in self._adj[u].keys():
                    if v in seen:
                        continue
                    seen.add(v)
                    stack.append(v)

            comps.append(comp^)

        return comps^

    fn number_connected_components(ref self) raises -> Int:
        return len(self.connected_components())

    fn is_connected(ref self) raises -> Bool:
        if self.number_of_nodes() == 0:
            return True
        return self.number_connected_components() == 1

    fn bfs_edges(ref self, source: Self.N) raises -> List[Tuple[Self.N, Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var out = List[Tuple[Self.N, Self.N]]()
        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            for v in self._adj[u].keys():
                if v in seen:
                    continue
                seen.add(v)
                queue.append(v)
                out.append((u, v))

        return out^

    fn dfs_edges(ref self, source: Self.N) raises -> List[Tuple[Self.N, Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var out = List[Tuple[Self.N, Self.N]]()
        var seen = Set[Self.N]()
        seen.add(source)

        var stack = List[Self.N]()
        stack.append(source)

        while len(stack) > 0:
            var u = stack.pop()
            for v in self._adj[u].keys():
                if v in seen:
                    continue
                seen.add(v)
                stack.append(v)
                out.append((u, v))

        return out^

    fn bfs_tree(ref self, source: Self.N, out t: Graph[Self.N]) raises:
        if not self.has_node(source):
            raise Error("node not in graph")

        t = Graph[Self.N]()
        t.add_node(source)

        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            for e in self._adj[u].items():
                var v = e.key
                if v in seen:
                    continue
                seen.add(v)
                queue.append(v)
                t.add_edge(u, v, e.value)

        return

    fn dfs_tree(ref self, source: Self.N, out t: Graph[Self.N]) raises:
        if not self.has_node(source):
            raise Error("node not in graph")

        t = Graph[Self.N]()
        t.add_node(source)

        var seen = Set[Self.N]()
        seen.add(source)

        var stack = List[Self.N]()
        stack.append(source)

        while len(stack) > 0:
            var u = stack.pop()
            for e in self._adj[u].items():
                var v = e.key
                if v in seen:
                    continue
                seen.add(v)
                stack.append(v)
                t.add_edge(u, v, e.value)

        return

    fn bfs_predecessors(ref self, source: Self.N) raises -> List[Tuple[Self.N, Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var pred = Dict[Self.N, Self.N]()
        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            for v in self._adj[u].keys():
                if v in seen:
                    continue
                seen.add(v)
                pred[v] = u
                queue.append(v)

        var out = List[Tuple[Self.N, Self.N]]()
        for entry in pred.items():
            out.append((entry.key, entry.value))
        return out^

    fn bfs_successors(ref self, source: Self.N) raises -> List[Tuple[Self.N, List[Self.N]]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var succs = Dict[Self.N, List[Self.N]]()
        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            for v in self._adj[u].keys():
                if v in seen:
                    continue
                seen.add(v)
                ref out_list = succs.setdefault(u, List[Self.N]())
                out_list.append(v)
                queue.append(v)

        var out = List[Tuple[Self.N, List[Self.N]]]()
        for entry in succs.items():
            out.append((entry.key, entry.value.copy()))
        return out^

    fn dfs_predecessors(ref self, source: Self.N) raises -> List[Tuple[Self.N, Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var pred = Dict[Self.N, Self.N]()
        var seen = Set[Self.N]()
        seen.add(source)

        var stack = List[Self.N]()
        stack.append(source)

        while len(stack) > 0:
            var u = stack.pop()
            for v in self._adj[u].keys():
                if v in seen:
                    continue
                seen.add(v)
                pred[v] = u
                stack.append(v)

        var out = List[Tuple[Self.N, Self.N]]()
        for entry in pred.items():
            out.append((entry.key, entry.value))
        return out^

    fn dfs_successors(ref self, source: Self.N) raises -> List[Tuple[Self.N, List[Self.N]]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var succs = Dict[Self.N, List[Self.N]]()
        var seen = Set[Self.N]()
        seen.add(source)

        var stack = List[Self.N]()
        stack.append(source)

        while len(stack) > 0:
            var u = stack.pop()
            for v in self._adj[u].keys():
                if v in seen:
                    continue
                seen.add(v)
                ref out_list = succs.setdefault(u, List[Self.N]())
                out_list.append(v)
                stack.append(v)

        var out = List[Tuple[Self.N, List[Self.N]]]()
        for entry in succs.items():
            out.append((entry.key, entry.value.copy()))
        return out^

    fn bfs_layers(ref self, source: Self.N) raises -> List[List[Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var layers = List[List[Self.N]]()
        var seen = Set[Self.N]()
        seen.add(source)

        var cur = List[Self.N]()
        cur.append(source)

        while len(cur) > 0:
            layers.append(cur.copy())
            var nxt = List[Self.N]()
            for u in cur:
                for v in self._adj[u].keys():
                    if v in seen:
                        continue
                    seen.add(v)
                    nxt.append(v)
            cur = nxt^

        return layers^

    fn minimum_spanning_tree(ref self, out mst: Graph[Self.N]) raises:
        mst = Graph[Self.N]()
        for node in self._adj.keys():
            mst.add_node(node)

        var uf = _UnionFind[Self.N]()
        for node in self._adj.keys():
            uf.add(node)

        var edges = List[_WEdge[Self.N]]()
        var processed = Set[Self.N]()
        for entry in self._adj.items():
            for e in entry.value.items():
                var v = e.key
                if entry.key == v:
                    continue
                if v in processed:
                    continue
                edges.append(_WEdge[Self.N](entry.key, v, e.value))
            processed.add(entry.key)

        _sort_wedges_in_place(edges)

        for e in edges:
            if uf.union(e.u, e.v):
                mst.add_edge(e.u, e.v, e.w)

        return

    fn shortest_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return [source]

        var visited = Set[Self.N]()
        visited.add(source)
        var parents = Dict[Self.N, Self.N]()

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            for v in self._adj[u].keys():
                if v in visited:
                    continue
                visited.add(v)
                parents[v] = u
                if v == target:
                    return self._reconstruct_path(parents, source, target)
                queue.append(v)

        raise Error("no path")

    fn bidirectional_shortest_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return [source]

        var visited_fwd = Set[Self.N]()
        var visited_bwd = Set[Self.N]()
        visited_fwd.add(source)
        visited_bwd.add(target)

        var parents_fwd = Dict[Self.N, Self.N]()
        var parents_bwd = Dict[Self.N, Self.N]()

        var frontier_fwd = List[Self.N]()
        var frontier_bwd = List[Self.N]()
        frontier_fwd.append(source)
        frontier_bwd.append(target)

        var meet = source
        var found = False

        while len(frontier_fwd) > 0 and len(frontier_bwd) > 0:
            if len(frontier_fwd) <= len(frontier_bwd):
                var next = List[Self.N]()
                for u in frontier_fwd:
                    for v in self._adj[u].keys():
                        if v in visited_fwd:
                            continue
                        visited_fwd.add(v)
                        parents_fwd[v] = u
                        if v in visited_bwd:
                            meet = v
                            found = True
                            break
                        next.append(v)
                    if found:
                        break
                if found:
                    break
                frontier_fwd = next^
            else:
                var next = List[Self.N]()
                for u in frontier_bwd:
                    for v in self._adj[u].keys():
                        if v in visited_bwd:
                            continue
                        visited_bwd.add(v)
                        parents_bwd[v] = u
                        if v in visited_fwd:
                            meet = v
                            found = True
                            break
                        next.append(v)
                    if found:
                        break
                if found:
                    break
                frontier_bwd = next^

        if not found:
            raise Error("no path")

        var path = self._reconstruct_path(parents_fwd, source, meet)
        var cur = meet
        while cur != target:
            cur = parents_bwd[cur]
            path.append(cur)
        return path^

    fn shortest_path_length(ref self, source: Self.N, target: Self.N) raises -> Int:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return 0

        var dist = Dict[Self.N, Int]()
        dist[source] = 0
        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            var du = dist[u]
            for v in self._adj[u].keys():
                if v in seen:
                    continue
                seen.add(v)
                dist[v] = du + 1
                if v == target:
                    return dist[v]
                queue.append(v)

        raise Error("no path")

    fn single_source_shortest_path_length(ref self, source: Self.N) raises -> Dict[Self.N, Int]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var dist = Dict[Self.N, Int]()
        dist[source] = 0
        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            var du = dist[u]
            for v in self._adj[u].keys():
                if v in seen:
                    continue
                seen.add(v)
                dist[v] = du + 1
                queue.append(v)

        return dist^

    fn single_source_shortest_path(ref self, source: Self.N) raises -> Dict[Self.N, List[Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var parents = Dict[Self.N, Self.N]()
        var dist = Dict[Self.N, Int]()
        dist[source] = 0
        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            var du = dist[u]
            for v in self._adj[u].keys():
                if v in seen:
                    continue
                seen.add(v)
                parents[v] = u
                dist[v] = du + 1
                queue.append(v)

        var paths = Dict[Self.N, List[Self.N]]()
        for node in dist.keys():
            if node == source:
                paths[node] = [source]
            else:
                paths[node] = self._reconstruct_path(parents, source, node)
        return paths^

    fn has_path(ref self, source: Self.N, target: Self.N) raises -> Bool:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return True

        var visited = Set[Self.N]()
        visited.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            for v in self._adj[u].keys():
                if v in visited:
                    continue
                if v == target:
                    return True
                visited.add(v)
                queue.append(v)

        return False

    fn all_pairs_shortest_path_length(ref self) raises -> Dict[Self.N, Dict[Self.N, Int]]:
        var out = Dict[Self.N, Dict[Self.N, Int]]()
        for entry in self._adj.items():
            var src = entry.key
            var dist = self.single_source_shortest_path_length(src)
            out[src] = dist^
        return out^

    fn all_pairs_shortest_path(ref self) raises -> Dict[Self.N, Dict[Self.N, List[Self.N]]]:
        var out = Dict[Self.N, Dict[Self.N, List[Self.N]]]()
        for entry in self._adj.items():
            var src = entry.key
            var paths = self.single_source_shortest_path(src)
            out[src] = paths^
        return out^

    fn floyd_warshall(ref self) raises -> Dict[Self.N, Dict[Self.N, Float64]]:
        var nodes = self.nodes()
        var n = len(nodes)
        var INF = 1.0e308

        var dist = Dict[Self.N, Dict[Self.N, Float64]]()
        var i = 0
        while i < n:
            var u = nodes[i]
            var row = Dict[Self.N, Float64]()
            var j = 0
            while j < n:
                row[nodes[j]] = INF
                j += 1
            row[u] = 0.0
            dist[u] = row^
            i += 1

        for entry in self._adj.items():
            var u = entry.key
            ref row_u = dist[u]
            for e in entry.value.items():
                var v = e.key
                var w = e.value
                if w < row_u[v]:
                    row_u[v] = w

        var k = 0
        while k < n:
            var kk = nodes[k]
            var i2 = 0
            while i2 < n:
                var ii = nodes[i2]
                ref row_i = dist[ii]
                var dik = row_i[kk]
                if dik >= INF:
                    i2 += 1
                    continue
                ref row_k = dist[kk]
                var j2 = 0
                while j2 < n:
                    var jj = nodes[j2]
                    var dkj = row_k[jj]
                    if dkj >= INF:
                        j2 += 1
                        continue
                    var nd = dik + dkj
                    if nd < row_i[jj]:
                        row_i[jj] = nd
                    j2 += 1
                i2 += 1
            k += 1

        return dist^

    fn floyd_warshall_predecessor_and_distance(ref self) raises -> Tuple[Dict[Self.N, Dict[Self.N, Self.N]], Dict[Self.N, Dict[Self.N, Float64]]]:
        var nodes = self.nodes()
        var n = len(nodes)
        var INF = 1.0e308

        var dist = Dict[Self.N, Dict[Self.N, Float64]]()
        var pred = Dict[Self.N, Dict[Self.N, Self.N]]()

        var i = 0
        while i < n:
            var u = nodes[i]
            var drow = Dict[Self.N, Float64]()
            var prow = Dict[Self.N, Self.N]()
            var j = 0
            while j < n:
                drow[nodes[j]] = INF
                j += 1
            drow[u] = 0.0
            prow[u] = u
            dist[u] = drow^
            pred[u] = prow^
            i += 1

        for entry in self._adj.items():
            var u = entry.key
            ref drow_u = dist[u]
            ref prow_u = pred[u]
            for e in entry.value.items():
                var v = e.key
                var w = e.value
                if w < drow_u[v]:
                    drow_u[v] = w
                    prow_u[v] = u

        var k = 0
        while k < n:
            var kk = nodes[k]
            var i2 = 0
            while i2 < n:
                var ii = nodes[i2]
                ref drow_i = dist[ii]
                var dik = drow_i[kk]
                if dik >= INF:
                    i2 += 1
                    continue
                ref drow_k = dist[kk]
                ref prow_k = pred[kk]
                ref prow_i = pred[ii]
                var j2 = 0
                while j2 < n:
                    var jj = nodes[j2]
                    var dkj = drow_k[jj]
                    if dkj >= INF:
                        j2 += 1
                        continue
                    var nd = dik + dkj
                    if nd < drow_i[jj]:
                        drow_i[jj] = nd
                        try:
                            prow_i[jj] = prow_k[jj]
                        except:
                            prow_i[jj] = kk
                    j2 += 1
                i2 += 1
            k += 1

        return (pred^, dist^)

    fn dijkstra_path_length(ref self, source: Self.N, target: Self.N) raises -> Float64:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return 0.0

        var dist = Dict[Self.N, Float64]()
        dist[source] = 0.0
        var finalized = Set[Self.N]()

        var heap = _MinHeap[Self.N]()
        var push_count = 0
        heap.push(_HeapItem[Self.N](0.0, push_count, source))
        push_count += 1

        while not heap.is_empty():
            var item = heap.pop_min()
            var u = item.node
            if u in finalized:
                continue
            finalized.add(u)
            if u == target:
                break

            var du = dist[u]
            for e in self._adj[u].items():
                var v = e.key
                if v in finalized:
                    continue
                var nd = du + e.value
                var better: Bool
                try:
                    better = nd < dist[v]
                except:
                    better = True
                if better:
                    dist[v] = nd
                    heap.push(_HeapItem[Self.N](nd, push_count, v))
                    push_count += 1

        if not (target in dist):
            raise Error("no path")
        return dist[target]

    fn single_source_dijkstra_path_length(ref self, source: Self.N) raises -> Dict[Self.N, Float64]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var dist = Dict[Self.N, Float64]()
        dist[source] = 0.0
        var finalized = Set[Self.N]()

        var heap = _MinHeap[Self.N]()
        var push_count = 0
        heap.push(_HeapItem[Self.N](0.0, push_count, source))
        push_count += 1

        while not heap.is_empty():
            var item = heap.pop_min()
            var u = item.node
            if u in finalized:
                continue
            finalized.add(u)

            var du = dist[u]
            for e in self._adj[u].items():
                var v = e.key
                if v in finalized:
                    continue
                var nd = du + e.value
                var better: Bool
                try:
                    better = nd < dist[v]
                except:
                    better = True
                if better:
                    dist[v] = nd
                    heap.push(_HeapItem[Self.N](nd, push_count, v))
                    push_count += 1

        return dist^

    fn single_source_dijkstra_path(ref self, source: Self.N) raises -> Dict[Self.N, List[Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var dist = Dict[Self.N, Float64]()
        dist[source] = 0.0
        var parents = Dict[Self.N, Self.N]()
        var finalized = Set[Self.N]()

        var heap = _MinHeap[Self.N]()
        var push_count = 0
        heap.push(_HeapItem[Self.N](0.0, push_count, source))
        push_count += 1

        while not heap.is_empty():
            var item = heap.pop_min()
            var u = item.node
            if u in finalized:
                continue
            finalized.add(u)

            var du = dist[u]
            for e in self._adj[u].items():
                var v = e.key
                if v in finalized:
                    continue
                var nd = du + e.value
                var better: Bool
                try:
                    better = nd < dist[v]
                except:
                    better = True
                if better:
                    dist[v] = nd
                    parents[v] = u
                    heap.push(_HeapItem[Self.N](nd, push_count, v))
                    push_count += 1

        var paths = Dict[Self.N, List[Self.N]]()
        for node in dist.keys():
            if node == source:
                paths[node] = [source]
            else:
                paths[node] = self._reconstruct_path(parents, source, node)
        return paths^

    fn multi_source_dijkstra_path_length(ref self, sources: List[Self.N]) raises -> Dict[Self.N, Float64]:
        if len(sources) == 0:
            raise Error("sources must not be empty")

        var dist = Dict[Self.N, Float64]()
        var finalized = Set[Self.N]()

        var heap = _MinHeap[Self.N]()
        var push_count = 0
        for s in sources:
            if not self.has_node(s):
                raise Error("node not in graph")
            dist[s] = 0.0
            heap.push(_HeapItem[Self.N](0.0, push_count, s))
            push_count += 1

        while not heap.is_empty():
            var item = heap.pop_min()
            var u = item.node
            if u in finalized:
                continue
            finalized.add(u)

            var du = dist[u]
            for e in self._adj[u].items():
                var v = e.key
                if v in finalized:
                    continue
                var nd = du + e.value
                var better: Bool
                try:
                    better = nd < dist[v]
                except:
                    better = True
                if better:
                    dist[v] = nd
                    heap.push(_HeapItem[Self.N](nd, push_count, v))
                    push_count += 1

        return dist^

    fn multi_source_dijkstra_path(ref self, sources: List[Self.N]) raises -> Dict[Self.N, List[Self.N]]:
        if len(sources) == 0:
            raise Error("sources must not be empty")

        var dist = Dict[Self.N, Float64]()
        var parents = Dict[Self.N, Self.N]()
        var root = Dict[Self.N, Self.N]()
        var finalized = Set[Self.N]()

        var heap = _MinHeap[Self.N]()
        var push_count = 0
        for s in sources:
            if not self.has_node(s):
                raise Error("node not in graph")
            dist[s] = 0.0
            root[s] = s
            heap.push(_HeapItem[Self.N](0.0, push_count, s))
            push_count += 1

        while not heap.is_empty():
            var item = heap.pop_min()
            var u = item.node
            if u in finalized:
                continue
            finalized.add(u)

            var du = dist[u]
            var src_u = root[u]
            for e in self._adj[u].items():
                var v = e.key
                if v in finalized:
                    continue
                var nd = du + e.value
                var better: Bool
                try:
                    better = nd < dist[v]
                except:
                    better = True
                if better:
                    dist[v] = nd
                    parents[v] = u
                    root[v] = src_u
                    heap.push(_HeapItem[Self.N](nd, push_count, v))
                    push_count += 1

        var paths = Dict[Self.N, List[Self.N]]()
        for node in dist.keys():
            var src = root[node]
            if node == src:
                paths[node] = [src]
            else:
                paths[node] = self._reconstruct_path(parents, src, node)
        return paths^

    fn dijkstra_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return [source]

        var dist = Dict[Self.N, Float64]()
        dist[source] = 0.0
        var parents = Dict[Self.N, Self.N]()
        var finalized = Set[Self.N]()

        var heap = _MinHeap[Self.N]()
        var push_count = 0
        heap.push(_HeapItem[Self.N](0.0, push_count, source))
        push_count += 1

        while not heap.is_empty():
            var item = heap.pop_min()
            var u = item.node
            if u in finalized:
                continue
            finalized.add(u)
            if u == target:
                break

            var du = dist[u]
            for e in self._adj[u].items():
                var v = e.key
                if v in finalized:
                    continue
                var nd = du + e.value
                var better: Bool
                try:
                    better = nd < dist[v]
                except:
                    better = True
                if better:
                    dist[v] = nd
                    parents[v] = u
                    heap.push(_HeapItem[Self.N](nd, push_count, v))
                    push_count += 1

        if not (target in dist):
            raise Error("no path")
        return self._reconstruct_path(parents, source, target)

    fn bidirectional_dijkstra_path_length(ref self, source: Self.N, target: Self.N) raises -> Float64:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return 0.0

        var INF = 1.0e308
        var dist_fwd = Dict[Self.N, Float64]()
        var dist_bwd = Dict[Self.N, Float64]()
        dist_fwd[source] = 0.0
        dist_bwd[target] = 0.0

        var parents_fwd = Dict[Self.N, Self.N]()
        var parents_bwd = Dict[Self.N, Self.N]()
        var finalized_fwd = Set[Self.N]()
        var finalized_bwd = Set[Self.N]()

        var heap_fwd = _MinHeap[Self.N]()
        var heap_bwd = _MinHeap[Self.N]()
        var push_count = 0
        heap_fwd.push(_HeapItem[Self.N](0.0, push_count, source))
        push_count += 1
        heap_bwd.push(_HeapItem[Self.N](0.0, push_count, target))
        push_count += 1

        var best = INF
        var meet = source
        var toggle = False

        while not heap_fwd.is_empty() or not heap_bwd.is_empty():
            var do_bwd = False
            if heap_fwd.is_empty():
                do_bwd = True
            elif heap_bwd.is_empty():
                do_bwd = False
            else:
                do_bwd = toggle

            if do_bwd:
                var item = heap_bwd.pop_min()
                var u = item.node
                if u in finalized_bwd:
                    toggle = not toggle
                    continue
                finalized_bwd.add(u)
                var du = dist_bwd[u]

                try:
                    var cand = du + dist_fwd[u]
                    if cand < best:
                        best = cand
                        meet = u
                except:
                    pass

                if du < best:
                    for e in self._adj[u].items():
                        var v = e.key
                        if v in finalized_bwd:
                            continue
                        var nd = du + e.value
                        var better: Bool
                        try:
                            better = nd < dist_bwd[v]
                        except:
                            better = True
                        if better:
                            dist_bwd[v] = nd
                            parents_bwd[v] = u
                            heap_bwd.push(_HeapItem[Self.N](nd, push_count, v))
                            push_count += 1
                            try:
                                var cand2 = nd + dist_fwd[v]
                                if cand2 < best:
                                    best = cand2
                                    meet = v
                            except:
                                pass
            else:
                var item = heap_fwd.pop_min()
                var u = item.node
                if u in finalized_fwd:
                    toggle = not toggle
                    continue
                finalized_fwd.add(u)
                var du = dist_fwd[u]

                try:
                    var cand = du + dist_bwd[u]
                    if cand < best:
                        best = cand
                        meet = u
                except:
                    pass

                if du < best:
                    for e in self._adj[u].items():
                        var v = e.key
                        if v in finalized_fwd:
                            continue
                        var nd = du + e.value
                        var better: Bool
                        try:
                            better = nd < dist_fwd[v]
                        except:
                            better = True
                        if better:
                            dist_fwd[v] = nd
                            parents_fwd[v] = u
                            heap_fwd.push(_HeapItem[Self.N](nd, push_count, v))
                            push_count += 1
                            try:
                                var cand2 = nd + dist_bwd[v]
                                if cand2 < best:
                                    best = cand2
                                    meet = v
                            except:
                                pass

            toggle = not toggle

        if best >= INF:
            raise Error("no path")
        return best

    fn bidirectional_dijkstra_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return [source]

        var INF = 1.0e308
        var dist_fwd = Dict[Self.N, Float64]()
        var dist_bwd = Dict[Self.N, Float64]()
        dist_fwd[source] = 0.0
        dist_bwd[target] = 0.0

        var parents_fwd = Dict[Self.N, Self.N]()
        var parents_bwd = Dict[Self.N, Self.N]()
        var finalized_fwd = Set[Self.N]()
        var finalized_bwd = Set[Self.N]()

        var heap_fwd = _MinHeap[Self.N]()
        var heap_bwd = _MinHeap[Self.N]()
        var push_count = 0
        heap_fwd.push(_HeapItem[Self.N](0.0, push_count, source))
        push_count += 1
        heap_bwd.push(_HeapItem[Self.N](0.0, push_count, target))
        push_count += 1

        var best = INF
        var meet = source
        var toggle = False

        while not heap_fwd.is_empty() or not heap_bwd.is_empty():
            var do_bwd = False
            if heap_fwd.is_empty():
                do_bwd = True
            elif heap_bwd.is_empty():
                do_bwd = False
            else:
                do_bwd = toggle

            if do_bwd:
                var item = heap_bwd.pop_min()
                var u = item.node
                if u in finalized_bwd:
                    toggle = not toggle
                    continue
                finalized_bwd.add(u)
                var du = dist_bwd[u]

                try:
                    var cand = du + dist_fwd[u]
                    if cand < best:
                        best = cand
                        meet = u
                except:
                    pass

                if du < best:
                    for e in self._adj[u].items():
                        var v = e.key
                        if v in finalized_bwd:
                            continue
                        var nd = du + e.value
                        var better: Bool
                        try:
                            better = nd < dist_bwd[v]
                        except:
                            better = True
                        if better:
                            dist_bwd[v] = nd
                            parents_bwd[v] = u
                            heap_bwd.push(_HeapItem[Self.N](nd, push_count, v))
                            push_count += 1
                            try:
                                var cand2 = nd + dist_fwd[v]
                                if cand2 < best:
                                    best = cand2
                                    meet = v
                            except:
                                pass
            else:
                var item = heap_fwd.pop_min()
                var u = item.node
                if u in finalized_fwd:
                    toggle = not toggle
                    continue
                finalized_fwd.add(u)
                var du = dist_fwd[u]

                try:
                    var cand = du + dist_bwd[u]
                    if cand < best:
                        best = cand
                        meet = u
                except:
                    pass

                if du < best:
                    for e in self._adj[u].items():
                        var v = e.key
                        if v in finalized_fwd:
                            continue
                        var nd = du + e.value
                        var better: Bool
                        try:
                            better = nd < dist_fwd[v]
                        except:
                            better = True
                        if better:
                            dist_fwd[v] = nd
                            parents_fwd[v] = u
                            heap_fwd.push(_HeapItem[Self.N](nd, push_count, v))
                            push_count += 1
                            try:
                                var cand2 = nd + dist_bwd[v]
                                if cand2 < best:
                                    best = cand2
                                    meet = v
                            except:
                                pass

            toggle = not toggle

        if best >= INF:
            raise Error("no path")

        var path: List[Self.N]
        if meet == source:
            path = [source]
        else:
            path = self._reconstruct_path(parents_fwd, source, meet)

        var cur = meet
        while cur != target:
            cur = parents_bwd[cur]
            path.append(cur)
        return path^

    fn dijkstra_path_weighted[weight_fn: fn(Self.N, Self.N) -> Float64](ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return [source]

        var dist = Dict[Self.N, Float64]()
        dist[source] = 0.0
        var parents = Dict[Self.N, Self.N]()
        var finalized = Set[Self.N]()

        var heap = _MinHeap[Self.N]()
        var push_count = 0
        heap.push(_HeapItem[Self.N](0.0, push_count, source))
        push_count += 1

        while not heap.is_empty():
            var item = heap.pop_min()
            var u = item.node
            if u in finalized:
                continue
            finalized.add(u)
            if u == target:
                break

            var du = dist[u]
            for v in self._adj[u].keys():
                if v in finalized:
                    continue
                var nd = du + weight_fn(u, v)
                var better: Bool
                try:
                    better = nd < dist[v]
                except:
                    better = True
                if better:
                    dist[v] = nd
                    parents[v] = u
                    heap.push(_HeapItem[Self.N](nd, push_count, v))
                    push_count += 1

        if not (target in dist):
            raise Error("no path")
        return self._reconstruct_path(parents, source, target)

    fn astar_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return [source]

        var gscore = Dict[Self.N, Float64]()
        gscore[source] = 0.0
        var parents = Dict[Self.N, Self.N]()
        var closed = Set[Self.N]()

        var heap = _MinHeap[Self.N]()
        var push_count = 0
        heap.push(_HeapItem[Self.N](0.0, push_count, source))
        push_count += 1

        while not heap.is_empty():
            var item = heap.pop_min()
            var u = item.node
            if u in closed:
                continue
            if u == target:
                return self._reconstruct_path(parents, source, target)
            closed.add(u)

            var gu = gscore[u]
            for e in self._adj[u].items():
                var v = e.key
                if v in closed:
                    continue
                var tentative = gu + e.value
                var better: Bool
                try:
                    better = tentative < gscore[v]
                except:
                    better = True
                if better:
                    gscore[v] = tentative
                    parents[v] = u
                    heap.push(_HeapItem[Self.N](tentative, push_count, v))
                    push_count += 1

        raise Error("no path")

    fn astar_path_weighted[weight_fn: fn(Self.N, Self.N) -> Float64, heuristic_fn: fn(Self.N, Self.N) -> Float64](ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
        if not self.has_node(source) or not self.has_node(target):
            raise Error("node not in graph")
        if source == target:
            return [source]

        var gscore = Dict[Self.N, Float64]()
        gscore[source] = 0.0
        var parents = Dict[Self.N, Self.N]()
        var closed = Set[Self.N]()

        var heap = _MinHeap[Self.N]()
        var push_count = 0
        heap.push(_HeapItem[Self.N](heuristic_fn(source, target), push_count, source))
        push_count += 1

        while not heap.is_empty():
            var item = heap.pop_min()
            var u = item.node
            if u in closed:
                continue
            if u == target:
                return self._reconstruct_path(parents, source, target)
            closed.add(u)

            var gu = gscore[u]
            for v in self._adj[u].keys():
                if v in closed:
                    continue
                var tentative = gu + weight_fn(u, v)
                var better: Bool
                try:
                    better = tentative < gscore[v]
                except:
                    better = True
                if better:
                    gscore[v] = tentative
                    parents[v] = u
                    var f = tentative + heuristic_fn(v, target)
                    heap.push(_HeapItem[Self.N](f, push_count, v))
                    push_count += 1

        raise Error("no path")

    fn has_node(self, node: Self.N) -> Bool:
        return node in self._adj

    fn nodes(ref self) -> List[Self.N]:
        var result = List[Self.N]()
        for node in self._adj.keys():
            result.append(node)
        return result^

    fn adj_view(ref self) -> ref[self._adj] Dict[Self.N, Dict[Self.N, Float64]]:
        return self._adj

    fn adj(ref self) -> ref[self._adj] Dict[Self.N, Dict[Self.N, Float64]]:
        return self._adj

    fn __getitem__(ref self, node: Self.N) raises -> Dict[Self.N, Float64]:
        return self._adj[node].copy()

    fn for_each_node[callback: fn(Self.N)](ref self) -> Int:
        var count = 0
        for node in self._adj.keys():
            callback(node)
            count += 1
        return count

    fn adj(ref self, node: Self.N) raises -> List[Self.N]:
        return self.neighbors(node)

    fn neighbors(ref self, node: Self.N) raises -> List[Self.N]:
        var result = List[Self.N]()
        for nbr in self._adj[node].keys():
            result.append(nbr)
        return result^

    fn for_each_neighbor[callback: fn(Self.N)](ref self, node: Self.N) raises -> Int:
        var count = 0
        for nbr in self._adj[node].keys():
            callback(nbr)
            count += 1
        return count

    fn add_node(mut self, node: Self.N):
        _ = self._adj.setdefault(node, Dict[Self.N, Float64]())
        _ = self._node_attr.setdefault(node, Dict[String, AttrValue]())
        _ = self._edge_attr.setdefault(node, Dict[Self.N, Dict[String, AttrValue]]())

    fn add_nodes_from(mut self, nodes: List[Self.N]):
        for node in nodes:
            self.add_node(node)

    fn add_edge(mut self, u: Self.N, v: Self.N, weight: Float64 = 1.0):
        ref neighbors_u = self._adj.setdefault(u, Dict[Self.N, Float64]())
        ref neighbors_v = self._adj.setdefault(v, Dict[Self.N, Float64]())

        neighbors_u[v] = weight
        if u != v:
            neighbors_v[u] = weight

        _ = self._node_attr.setdefault(u, Dict[String, AttrValue]())
        _ = self._node_attr.setdefault(v, Dict[String, AttrValue]())
        ref u_edges = self._edge_attr.setdefault(u, Dict[Self.N, Dict[String, AttrValue]]())
        ref v_edges = self._edge_attr.setdefault(v, Dict[Self.N, Dict[String, AttrValue]]())

        ref u_map = u_edges.setdefault(v, Dict[String, AttrValue]())
        u_map["weight"] = AttrValue(weight)
        if u != v:
            ref v_map = v_edges.setdefault(u, Dict[String, AttrValue]())
            v_map["weight"] = AttrValue(weight)

    fn add_edges_from(mut self, edges: List[Tuple[Self.N, Self.N]]):
        for e in edges:
            self.add_edge(e[0], e[1])

    fn copy(ref self, out g: Graph[Self.N]) raises:
        g = Graph[Self.N]()

        for node in self._adj.keys():
            g.add_node(node)

        var processed = Set[Self.N]()
        for entry in self._adj.items():
            for nbr_entry in entry.value.items():
                var v = nbr_entry.key
                if v in processed:
                    continue
                g.add_edge(entry.key, v, nbr_entry.value)
            processed.add(entry.key)

        for entry in self._graph_attr.items():
            g.set_graph_attr(entry.key, entry.value)

        for entry in self._node_attr.items():
            for kv in entry.value.items():
                g.set_node_attr(entry.key, kv.key, kv.value)

        var processed_edges = Set[Self.N]()
        for u_entry in self._edge_attr.items():
            for v_entry in u_entry.value.items():
                var v = v_entry.key
                if v in processed_edges:
                    continue
                for kv in v_entry.value.items():
                    if kv.key == "weight":
                        continue
                    var tmp = kv.value
                    g.set_edge_attr(u_entry.key, v, kv.key, tmp)
            processed_edges.add(u_entry.key)

        return

    fn subgraph(ref self, nodes: List[Self.N], out sg: Graph[Self.N]) raises:
        sg = Graph[Self.N]()
        var node_set = Set[Self.N]()

        for n in nodes:
            if self.has_node(n):
                sg.add_node(n)
                node_set.add(n)

        for entry in self._graph_attr.items():
            sg.set_graph_attr(entry.key, entry.value)

        for entry in self._node_attr.items():
            if not (entry.key in node_set):
                continue
            for kv in entry.value.items():
                sg.set_node_attr(entry.key, kv.key, kv.value)

        var processed = Set[Self.N]()
        for entry in self._adj.items():
            if not (entry.key in node_set):
                continue
            for nbr_entry in entry.value.items():
                var v = nbr_entry.key
                if not (v in node_set):
                    continue
                if v in processed:
                    continue
                sg.add_edge(entry.key, v, nbr_entry.value)
            processed.add(entry.key)

        var processed_edges = Set[Self.N]()
        for u_entry in self._edge_attr.items():
            if not (u_entry.key in node_set):
                continue
            for v_entry in u_entry.value.items():
                var v = v_entry.key
                if not (v in node_set):
                    continue
                if v in processed_edges:
                    continue
                for kv in v_entry.value.items():
                    if kv.key == "weight":
                        continue
                    var tmp = kv.value
                    sg.set_edge_attr(u_entry.key, v, kv.key, tmp)
            processed_edges.add(u_entry.key)

        return

    fn clear(mut self):
        self._adj = Dict[Self.N, Dict[Self.N, Float64]]()
        self._graph_attr = Dict[String, AttrValue]()
        self._node_attr = Dict[Self.N, Dict[String, AttrValue]]()
        self._edge_attr = Dict[Self.N, Dict[Self.N, Dict[String, AttrValue]]]()

    fn set_graph_attr(mut self, key: String, value: AttrValue):
        self._graph_attr[key] = value

    fn get_graph_attr(ref self, key: String) raises -> AttrValue:
        return self._graph_attr[key]

    fn set_node_attr(mut self, node: Self.N, key: String, value: AttrValue) raises:
        if not self.has_node(node):
            raise Error("node not in graph")
        ref attrs = self._node_attr.setdefault(node, Dict[String, AttrValue]())
        attrs[key] = value

    fn get_node_attr(ref self, node: Self.N, key: String) raises -> AttrValue:
        if not self.has_node(node):
            raise Error("node not in graph")
        return self._node_attr[node][key]

    fn set_edge_attr(mut self, u: Self.N, v: Self.N, key: String, mut value: AttrValue) raises:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")

        if key == "weight":
            if not value.isa[Float64]():
                raise Error("weight must be Float64")
            self.add_edge(u, v, value[Float64])
            return

        ref u_edges = self._edge_attr.setdefault(u, Dict[Self.N, Dict[String, AttrValue]]())
        ref v_edges = self._edge_attr.setdefault(v, Dict[Self.N, Dict[String, AttrValue]]())

        ref u_map = u_edges.setdefault(v, Dict[String, AttrValue]())
        u_map[key] = value
        if u != v:
            ref v_map = v_edges.setdefault(u, Dict[String, AttrValue]())
            v_map[key] = value

    fn get_edge_attr(ref self, u: Self.N, v: Self.N, key: String) raises -> AttrValue:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")
        if key == "weight":
            return AttrValue(self._adj[u][v])
        return self._edge_attr[u][v][key]

    fn degree(self, node: Self.N) raises -> Int:
        var d = len(self._adj[node])
        if node in self._adj[node]:
            d += 1
        return d

    fn has_edge(self, u: Self.N, v: Self.N) -> Bool:
        try:
            return v in self._adj[u]
        except:
            return False

    fn edges(self) -> List[Tuple[Self.N, Self.N]]:
        var result = List[Tuple[Self.N, Self.N]]()
        var processed = Set[Self.N]()

        for entry in self._adj.items():
            for nbr in entry.value.keys():
                if nbr in processed:
                    continue
                result.append((entry.key, nbr))
            processed.add(entry.key)
        return result^

    fn for_each_edge[callback: fn(Self.N, Self.N)](ref self) -> Int:
        var count = 0
        var processed = Set[Self.N]()

        for entry in self._adj.items():
            for nbr in entry.value.keys():
                if nbr in processed:
                    continue
                callback(entry.key, nbr)
                count += 1
            processed.add(entry.key)
        return count

    fn remove_edge(mut self, u: Self.N, v: Self.N) raises:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")

        _ = self._adj[u].pop(v)
        if u != v:
            _ = self._adj[v].pop(u)

        try:
            _ = self._edge_attr[u].pop(v)
        except:
            pass
        if u != v:
            try:
                _ = self._edge_attr[v].pop(u)
            except:
                pass

    fn remove_edges_from(mut self, edges: List[Tuple[Self.N, Self.N]]):
        for e in edges:
            if self.has_edge(e[0], e[1]):
                try:
                    self.remove_edge(e[0], e[1])
                except:
                    pass

    fn remove_node(mut self, node: Self.N) raises:
        var neighbors = self._adj.pop(node)
        for nbr in neighbors.keys():
            if nbr == node:
                continue
            _ = self._adj[nbr].pop(node)

        try:
            _ = self._node_attr.pop(node)
        except:
            pass

        try:
            _ = self._edge_attr.pop(node)
        except:
            pass

        for nbr in neighbors.keys():
            if nbr == node:
                continue
            try:
                _ = self._edge_attr[nbr].pop(node)
            except:
                pass

    fn remove_nodes_from(mut self, nodes: List[Self.N]):
        for n in nodes:
            if self.has_node(n):
                try:
                    self.remove_node(n)
                except:
                    pass
