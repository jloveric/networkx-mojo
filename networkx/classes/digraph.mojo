from std.traits.copyable import ImplicitlyCopyable
from std.traits import Deinitable
from std.collections import Dict, List, Set
from std.collections.dict import KeyElement
from std.utils import Variant
from .._internal.heap import _HeapItem, _MinHeap
from .._internal.dict_helpers import (
    dict_get_nn,
    dict_get_nf,
    dict_get_ni,
    dict_contains_key,
    dict_contains_nested_key,
    dict_neighbor_keys,
    dict_adj_entries,
    dict_set_inner_map,
    dict_get_nf_cell,
    dict_set_nf_cell,
    dict_get_nn_cell,
    dict_set_nn_cell,
)

def _unit_weight[N: KeyElement & Deinitable & ImplicitlyCopyable](u: N, v: N) -> Float64:
    return 1.0

def _zero_heuristic[N: KeyElement & Deinitable & ImplicitlyCopyable](u: N, v: N) -> Float64:
    return 0.0

def _reverse_in_place[N: KeyElement & Deinitable & ImplicitlyCopyable](mut path: List[N]):
    var i = 0
    var j = len(path) - 1
    while i < j:
        var tmp = path[i]
        path[i] = path[j]
        path[j] = tmp
        i += 1
        j -= 1

comptime AttrValue = Variant[Int, Float64, Bool, String]

struct DiGraph[N: KeyElement & Deinitable & ImplicitlyCopyable]:
    var _succ: Dict[Self.N, Dict[Self.N, Float64]]
    var _pred: Dict[Self.N, Dict[Self.N, Float64]]
    var _graph_attr: Dict[String, AttrValue]
    var _node_attr: Dict[Self.N, Dict[String, AttrValue]]
    var _edge_attr: Dict[Self.N, Dict[Self.N, Dict[String, AttrValue]]]

    def __init__(out self):
        self._succ = Dict[Self.N, Dict[Self.N, Float64]]()
        self._pred = Dict[Self.N, Dict[Self.N, Float64]]()
        self._graph_attr = Dict[String, AttrValue]()
        self._node_attr = Dict[Self.N, Dict[String, AttrValue]]()
        self._edge_attr = Dict[Self.N, Dict[Self.N, Dict[String, AttrValue]]]()

    def _reconstruct_path(ref self, ref parents: Dict[Self.N, Self.N], source: Self.N, target: Self.N) raises -> List[Self.N]:
        var path = List[Self.N]()
        var cur: Self.N = target
        path.append(cur)
        while cur != source:
            cur = dict_get_nn(parents, cur)
            path.append(cur)
        _reverse_in_place(path)
        return path^

    def __len__(self) -> Int:
        return self.number_of_nodes()

    def __contains__(self, node: Self.N) -> Bool:
        return self.has_node(node)

    def __iter__(self) -> Dict[Self.N, Dict[Self.N, Float64]].IteratorType[iterable_mut=False, iterable_origin=origin_of(self._succ)]:
        return self._succ.keys()

    def number_of_nodes(self) -> Int:
        return len(self._succ)

    def order(self) -> Int:
        return self.number_of_nodes()

    def number_of_edges(self) -> Int:
        var total_out_degree = 0
        for e in self._succ.items():
            total_out_degree += len(e.value)
        return total_out_degree

    def size(self) -> Int:
        return self.number_of_edges()

    def is_directed(self) -> Bool:
        return True

    def is_multigraph(self) -> Bool:
        return False

    def is_dag(ref self) -> Bool:
        try:
            _ = self.topological_sort()
            return True
        except:
            return False

    def bfs_edges(ref self, source: Self.N) raises -> List[Tuple[Self.N, Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var out = List[Tuple[Self.N, Self.N]]()
        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u: Self.N = queue[head]
            head += 1
            for v in dict_neighbor_keys(self._succ, u):
                if v in seen:
                    continue
                seen.add(v)
                queue.append(v)
                out.append((u, v))

        return out^

    def dfs_edges(ref self, source: Self.N) raises -> List[Tuple[Self.N, Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var out = List[Tuple[Self.N, Self.N]]()
        var seen = Set[Self.N]()
        seen.add(source)

        var stack = List[Self.N]()
        stack.append(source)

        while len(stack) > 0:
            var u: Self.N = stack.pop()
            for v in dict_neighbor_keys(self._succ, u):
                if v in seen:
                    continue
                seen.add(v)
                stack.append(v)
                out.append((u, v))

        return out^

    def bfs_tree(ref self, source: Self.N, out t: DiGraph[Self.N]) raises:
        if not self.has_node(source):
            raise Error("node not in graph")

        t = DiGraph[Self.N]()
        t.add_node(source)

        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u: Self.N = queue[head]
            head += 1
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                if v in seen:
                    continue
                seen.add(v)
                queue.append(v)
                t.add_edge(u, v, e[1])

        return

    def dfs_tree(ref self, source: Self.N, out t: DiGraph[Self.N]) raises:
        if not self.has_node(source):
            raise Error("node not in graph")

        t = DiGraph[Self.N]()
        t.add_node(source)

        var seen = Set[Self.N]()
        seen.add(source)

        var stack = List[Self.N]()
        stack.append(source)

        while len(stack) > 0:
            var u: Self.N = stack.pop()
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                if v in seen:
                    continue
                seen.add(v)
                stack.append(v)
                t.add_edge(u, v, e[1])

        return

    def bfs_predecessors(ref self, source: Self.N) raises -> List[Tuple[Self.N, Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var pred = Dict[Self.N, Self.N]()
        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u: Self.N = queue[head]
            head += 1
            for v in dict_neighbor_keys(self._succ, u):
                if v in seen:
                    continue
                seen.add(v)
                pred[v] = u
                queue.append(v)

        var out = List[Tuple[Self.N, Self.N]]()
        for entry in pred.items():
            out.append((entry.key, entry.value))
        return out^

    def bfs_successors(ref self, source: Self.N) raises -> List[Tuple[Self.N, List[Self.N]]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var succs = Dict[Self.N, List[Self.N]]()
        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u: Self.N = queue[head]
            head += 1
            for v in dict_neighbor_keys(self._succ, u):
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

    def dfs_predecessors(ref self, source: Self.N) raises -> List[Tuple[Self.N, Self.N]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var pred = Dict[Self.N, Self.N]()
        var seen = Set[Self.N]()
        seen.add(source)

        var stack = List[Self.N]()
        stack.append(source)

        while len(stack) > 0:
            var u: Self.N = stack.pop()
            for v in dict_neighbor_keys(self._succ, u):
                if v in seen:
                    continue
                seen.add(v)
                pred[v] = u
                stack.append(v)

        var out = List[Tuple[Self.N, Self.N]]()
        for entry in pred.items():
            out.append((entry.key, entry.value))
        return out^

    def dfs_successors(ref self, source: Self.N) raises -> List[Tuple[Self.N, List[Self.N]]]:
        if not self.has_node(source):
            raise Error("node not in graph")

        var succs = Dict[Self.N, List[Self.N]]()
        var seen = Set[Self.N]()
        seen.add(source)

        var stack = List[Self.N]()
        stack.append(source)

        while len(stack) > 0:
            var u: Self.N = stack.pop()
            for v in dict_neighbor_keys(self._succ, u):
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

    def bfs_layers(ref self, source: Self.N) raises -> List[List[Self.N]]:
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
                for v in dict_neighbor_keys(self._succ, u):
                    if v in seen:
                        continue
                    seen.add(v)
                    nxt.append(v)
            cur = nxt^

        return layers^

    def descendants(ref self, node: Self.N) raises -> List[Self.N]:
        if not self.has_node(node):
            raise Error("node not in graph")

        var seen = Set[Self.N]()
        var out = List[Self.N]()
        var stack = List[Self.N]()

        for v in self._succ[node].keys():
            stack.append(v)

        while len(stack) > 0:
            var u: Self.N = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
            for v in dict_neighbor_keys(self._succ, u):
                if not (v in seen):
                    stack.append(v)

        return out^

    def ancestors(ref self, node: Self.N) raises -> List[Self.N]:
        if not self.has_node(node):
            raise Error("node not in graph")

        var seen = Set[Self.N]()
        var out = List[Self.N]()
        var stack = List[Self.N]()

        for v in self._pred[node].keys():
            stack.append(v)

        while len(stack) > 0:
            var u: Self.N = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
            for v in dict_neighbor_keys(self._pred, u):
                if not (v in seen):
                    stack.append(v)

        return out^

    def topological_sort(ref self) raises -> List[Self.N]:
        var indeg = Dict[Self.N, Int]()
        for node in self._succ.keys():
            indeg[node] = 0

        for entry in self._succ.items():
            for v in entry.value.keys():
                indeg[v] = indeg[v] + 1

        var queue = List[Self.N]()
        for entry in indeg.items():
            if entry.value == 0:
                queue.append(entry.key)

        var out = List[Self.N]()
        var head = 0
        while head < len(queue):
            var u: Self.N = queue[head]
            head += 1
            out.append(u)

            for v in dict_neighbor_keys(self._succ, u):
                indeg[v] = indeg[v] - 1
                if indeg[v] == 0:
                    queue.append(v)

        if len(out) != len(indeg):
            raise Error("graph has a cycle")
        return out^

    def shortest_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
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
            var u: Self.N = queue[head]
            head += 1
            for v in dict_neighbor_keys(self._succ, u):
                if v in visited:
                    continue
                visited.add(v)
                parents[v] = u
                if v == target:
                    return self._reconstruct_path(parents, source, target)
                queue.append(v)

        raise Error("no path")

    def bidirectional_shortest_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
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
                    for v in dict_neighbor_keys(self._succ, u):
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
                    for v in self._pred[u].keys():
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
        var cur: Self.N = meet
        while cur != target:
            cur = dict_get_nn(parents_bwd, cur)
            path.append(cur)
        return path^

    def shortest_path_length(ref self, source: Self.N, target: Self.N) raises -> Int:
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
            var u: Self.N = queue[head]
            head += 1
            var du = dict_get_ni(dist, u)
            for v in dict_neighbor_keys(self._succ, u):
                if v in seen:
                    continue
                seen.add(v)
                dist[v] = du + 1
                if v == target:
                    return dist[v]
                queue.append(v)

        raise Error("no path")

    def single_source_shortest_path_length(ref self, source: Self.N) raises -> Dict[Self.N, Int]:
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
            var u: Self.N = queue[head]
            head += 1
            var du = dict_get_ni(dist, u)
            for v in dict_neighbor_keys(self._succ, u):
                if v in seen:
                    continue
                seen.add(v)
                dist[v] = du + 1
                queue.append(v)

        return dist^

    def single_source_shortest_path(ref self, source: Self.N) raises -> Dict[Self.N, List[Self.N]]:
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
            var u: Self.N = queue[head]
            head += 1
            var du = dict_get_ni(dist, u)
            for v in dict_neighbor_keys(self._succ, u):
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

    def has_path(ref self, source: Self.N, target: Self.N) raises -> Bool:
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
            var u: Self.N = queue[head]
            head += 1
            for v in dict_neighbor_keys(self._succ, u):
                if v in visited:
                    continue
                if v == target:
                    return True
                visited.add(v)
                queue.append(v)

        return False

    def all_pairs_shortest_path_length(ref self) raises -> Dict[Self.N, Dict[Self.N, Int]]:
        var out = Dict[Self.N, Dict[Self.N, Int]]()
        for entry in self._succ.items():
            var src: Self.N = entry.key
            var dist = self.single_source_shortest_path_length(src)
            dict_set_inner_map(out, src, dist^)
        return out^

    def all_pairs_shortest_path(ref self) raises -> Dict[Self.N, Dict[Self.N, List[Self.N]]]:
        var out = Dict[Self.N, Dict[Self.N, List[Self.N]]]()
        for entry in self._succ.items():
            var src: Self.N = entry.key
            var paths = self.single_source_shortest_path(src)
            dict_set_inner_map(out, src, paths^)
        return out^

    def floyd_warshall(ref self) raises -> Dict[Self.N, Dict[Self.N, Float64]]:
        var nodes = self.nodes()
        var n = len(nodes)
        var INF = 1.0e308

        var dist = Dict[Self.N, Dict[Self.N, Float64]]()
        var i = 0
        while i < n:
            var u: Self.N = nodes[i]
            var row = Dict[Self.N, Float64]()
            var j = 0
            while j < n:
                row[nodes[j]] = INF
                j += 1
            row[u] = 0.0
            dist[u] = row^
            i += 1

        for entry in self._succ.items():
            var u: Self.N = entry.key
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                var w = e[1]
                if w < dict_get_nf_cell(dist, u, v):
                    dict_set_nf_cell(dist, u, v, w)

        var k = 0
        while k < n:
            var kk = nodes[k]
            var i2 = 0
            while i2 < n:
                var ii = nodes[i2]
                var dik = dict_get_nf_cell(dist, ii, kk)
                if dik >= INF:
                    i2 += 1
                    continue
                var j2 = 0
                while j2 < n:
                    var jj = nodes[j2]
                    var dkj = dict_get_nf_cell(dist, kk, jj)
                    if dkj >= INF:
                        j2 += 1
                        continue
                    var nd = dik + dkj
                    if nd < dict_get_nf_cell(dist, ii, jj):
                        dict_set_nf_cell(dist, ii, jj, nd)
                    j2 += 1
                i2 += 1
            k += 1

        return dist^

    def floyd_warshall_predecessor_and_distance(ref self) raises -> Tuple[Dict[Self.N, Dict[Self.N, Self.N]], Dict[Self.N, Dict[Self.N, Float64]]]:
        var nodes = self.nodes()
        var n = len(nodes)
        var INF = 1.0e308

        var dist = Dict[Self.N, Dict[Self.N, Float64]]()
        var pred = Dict[Self.N, Dict[Self.N, Self.N]]()

        var i = 0
        while i < n:
            var u: Self.N = nodes[i]
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

        for entry in self._succ.items():
            var u: Self.N = entry.key
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                var w = e[1]
                if w < dict_get_nf_cell(dist, u, v):
                    dict_set_nf_cell(dist, u, v, w)
                    dict_set_nn_cell(pred, u, v, u)

        var k = 0
        while k < n:
            var kk = nodes[k]
            var i2 = 0
            while i2 < n:
                var ii = nodes[i2]
                var dik = dict_get_nf_cell(dist, ii, kk)
                if dik >= INF:
                    i2 += 1
                    continue
                var j2 = 0
                while j2 < n:
                    var jj = nodes[j2]
                    var dkj = dict_get_nf_cell(dist, kk, jj)
                    if dkj >= INF:
                        j2 += 1
                        continue
                    var nd = dik + dkj
                    if nd < dict_get_nf_cell(dist, ii, jj):
                        dict_set_nf_cell(dist, ii, jj, nd)
                        try:
                            dict_set_nn_cell(pred, ii, jj, dict_get_nn_cell(pred, kk, jj))
                        except:
                            dict_set_nn_cell(pred, ii, jj, kk)
                    j2 += 1
                i2 += 1
            k += 1

        return (pred^, dist^)

    def dijkstra_path_length(ref self, source: Self.N, target: Self.N) raises -> Float64:
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
            var u: Self.N = item.node
            if u in finalized:
                continue
            finalized.add(u)
            if u == target:
                break

            var du = dict_get_nf(dist, u)
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                if v in finalized:
                    continue
                var nd = du + e[1]
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

    def single_source_dijkstra_path_length(ref self, source: Self.N) raises -> Dict[Self.N, Float64]:
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
            var u: Self.N = item.node
            if u in finalized:
                continue
            finalized.add(u)

            var du = dict_get_nf(dist, u)
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                if v in finalized:
                    continue
                var nd = du + e[1]
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

    def single_source_dijkstra_path(ref self, source: Self.N) raises -> Dict[Self.N, List[Self.N]]:
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
            var u: Self.N = item.node
            if u in finalized:
                continue
            finalized.add(u)

            var du = dict_get_nf(dist, u)
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                if v in finalized:
                    continue
                var nd = du + e[1]
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

    def multi_source_dijkstra_path_length(ref self, sources: List[Self.N]) raises -> Dict[Self.N, Float64]:
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
            var u: Self.N = item.node
            if u in finalized:
                continue
            finalized.add(u)

            var du = dict_get_nf(dist, u)
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                if v in finalized:
                    continue
                var nd = du + e[1]
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

    def multi_source_dijkstra_path(ref self, sources: List[Self.N]) raises -> Dict[Self.N, List[Self.N]]:
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
            var u: Self.N = item.node
            if u in finalized:
                continue
            finalized.add(u)

            var du = dict_get_nf(dist, u)
            var src_u = root[u]
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                if v in finalized:
                    continue
                var nd = du + e[1]
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

    def dijkstra_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
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
            var u: Self.N = item.node
            if u in finalized:
                continue
            finalized.add(u)
            if u == target:
                break

            var du = dict_get_nf(dist, u)
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                if v in finalized:
                    continue
                var nd = du + e[1]
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

    def bidirectional_dijkstra_path_length(ref self, source: Self.N, target: Self.N) raises -> Float64:
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
        var toggle = False

        while not heap_fwd.is_empty() or not heap_bwd.is_empty():
            var do_bwd = toggle
            if heap_fwd.is_empty():
                do_bwd = True
            elif heap_bwd.is_empty():
                do_bwd = False

            if do_bwd:
                var item = heap_bwd.pop_min()
                var u: Self.N = item.node
                if u in finalized_bwd:
                    toggle = not toggle
                    continue
                finalized_bwd.add(u)
                var du = dict_get_nf(dist_bwd, u)

                try:
                    var cand = du + dist_fwd[u]
                    if cand < best:
                        best = cand
                except:
                    pass

                if du < best:
                    for e in dict_adj_entries(self._pred, u):
                        var v: Self.N = e[0]
                        if v in finalized_bwd:
                            continue
                        var nd = du + e[1]
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
                                var cand2 = nd + dict_get_nf(dist_fwd, v)
                                if cand2 < best:
                                    best = cand2
                            except:
                                pass
            else:
                var item = heap_fwd.pop_min()
                var u: Self.N = item.node
                if u in finalized_fwd:
                    toggle = not toggle
                    continue
                finalized_fwd.add(u)
                var du = dict_get_nf(dist_fwd, u)

                try:
                    var cand = du + dist_bwd[u]
                    if cand < best:
                        best = cand
                except:
                    pass

                if du < best:
                    for e in dict_adj_entries(self._succ, u):
                        var v: Self.N = e[0]
                        if v in finalized_fwd:
                            continue
                        var nd = du + e[1]
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
                                var cand2 = nd + dict_get_nf(dist_bwd, v)
                                if cand2 < best:
                                    best = cand2
                            except:
                                pass

            toggle = not toggle

        if best >= INF:
            raise Error("no path")
        return best

    def bidirectional_dijkstra_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
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
            var do_bwd = toggle
            if heap_fwd.is_empty():
                do_bwd = True
            elif heap_bwd.is_empty():
                do_bwd = False

            if do_bwd:
                var item = heap_bwd.pop_min()
                var u: Self.N = item.node
                if u in finalized_bwd:
                    toggle = not toggle
                    continue
                finalized_bwd.add(u)
                var du = dict_get_nf(dist_bwd, u)

                try:
                    var cand = du + dist_fwd[u]
                    if cand < best:
                        best = cand
                        meet = u
                except:
                    pass

                if du < best:
                    for e in dict_adj_entries(self._pred, u):
                        var v: Self.N = e[0]
                        if v in finalized_bwd:
                            continue
                        var nd = du + e[1]
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
                                var cand2 = nd + dict_get_nf(dist_fwd, v)
                                if cand2 < best:
                                    best = cand2
                                    meet = v
                            except:
                                pass
            else:
                var item = heap_fwd.pop_min()
                var u: Self.N = item.node
                if u in finalized_fwd:
                    toggle = not toggle
                    continue
                finalized_fwd.add(u)
                var du = dict_get_nf(dist_fwd, u)

                try:
                    var cand = du + dist_bwd[u]
                    if cand < best:
                        best = cand
                        meet = u
                except:
                    pass

                if du < best:
                    for e in dict_adj_entries(self._succ, u):
                        var v: Self.N = e[0]
                        if v in finalized_fwd:
                            continue
                        var nd = du + e[1]
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
                                var cand2 = nd + dict_get_nf(dist_bwd, v)
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

        var cur: Self.N = meet
        while cur != target:
            cur = dict_get_nn(parents_bwd, cur)
            path.append(cur)
        return path^

    def dijkstra_path_weighted[weight_fn: def(Self.N, Self.N) thin -> Float64](ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
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
            var u: Self.N = item.node
            if u in finalized:
                continue
            finalized.add(u)
            if u == target:
                break

            var du = dict_get_nf(dist, u)
            for v in dict_neighbor_keys(self._succ, u):
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

    def astar_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
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
            var u: Self.N = item.node
            if u in closed:
                continue
            if u == target:
                return self._reconstruct_path(parents, source, target)
            closed.add(u)

            var gu = gscore[u]
            for e in dict_adj_entries(self._succ, u):
                var v: Self.N = e[0]
                if v in closed:
                    continue
                var tentative = gu + e[1]
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

    def astar_path_weighted[weight_fn: def(Self.N, Self.N) thin -> Float64, heuristic_fn: def(Self.N, Self.N) thin -> Float64](ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
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
            var u: Self.N = item.node
            if u in closed:
                continue
            if u == target:
                return self._reconstruct_path(parents, source, target)
            closed.add(u)

            var gu = gscore[u]
            for v in dict_neighbor_keys(self._succ, u):
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

    def has_node(self, node: Self.N) -> Bool:
        return node in self._succ

    def nodes(ref self) -> List[Self.N]:
        var result = List[Self.N]()
        for node in self._succ.keys():
            result.append(node)
        return result^

    def succ_view(ref self) -> ref[self._succ] Dict[Self.N, Dict[Self.N, Float64]]:
        return self._succ

    def pred_view(ref self) -> ref[self._pred] Dict[Self.N, Dict[Self.N, Float64]]:
        return self._pred

    def succ(ref self) -> ref[self._succ] Dict[Self.N, Dict[Self.N, Float64]]:
        return self._succ

    def pred(ref self) -> ref[self._pred] Dict[Self.N, Dict[Self.N, Float64]]:
        return self._pred

    def adj(ref self) -> ref[self._succ] Dict[Self.N, Dict[Self.N, Float64]]:
        return self._succ

    def __getitem__(ref self, node: Self.N) raises -> Dict[Self.N, Float64]:
        return self._succ[node].copy()

    def for_each_node[callback: def(Self.N) thin -> None](ref self) -> Int:
        var count = 0
        for node in self._succ.keys():
            callback(node)
            count += 1
        return count

    def successors(ref self, node: Self.N) raises -> List[Self.N]:
        var result = List[Self.N]()
        for nbr in self._succ[node].keys():
            result.append(nbr)
        return result^

    def for_each_successor[callback: def(Self.N) thin -> None](ref self, node: Self.N) raises -> Int:
        var count = 0
        for nbr in self._succ[node].keys():
            callback(nbr)
            count += 1
        return count

    def neighbors(ref self, node: Self.N) raises -> List[Self.N]:
        return self.successors(node)

    def adj(ref self, node: Self.N) raises -> List[Self.N]:
        return self.successors(node)

    def predecessors(ref self, node: Self.N) raises -> List[Self.N]:
        var result = List[Self.N]()
        for nbr in self._pred[node].keys():
            result.append(nbr)
        return result^

    def for_each_predecessor[callback: def(Self.N) thin -> None](ref self, node: Self.N) raises -> Int:
        var count = 0
        for nbr in self._pred[node].keys():
            callback(nbr)
            count += 1
        return count

    def add_node(mut self, node: Self.N):
        _ = self._succ.setdefault(node, Dict[Self.N, Float64]())
        _ = self._pred.setdefault(node, Dict[Self.N, Float64]())
        _ = self._node_attr.setdefault(node, Dict[String, AttrValue]())
        _ = self._edge_attr.setdefault(node, Dict[Self.N, Dict[String, AttrValue]]())

    def add_nodes_from(mut self, nodes: List[Self.N]):
        for node in nodes:
            self.add_node(node)

    def add_edge(mut self, u: Self.N, v: Self.N, weight: Float64 = 1.0):
        ref succ_u = self._succ.setdefault(u, Dict[Self.N, Float64]())
        succ_u[v] = weight
        _ = self._pred.setdefault(u, Dict[Self.N, Float64]())

        _ = self._succ.setdefault(v, Dict[Self.N, Float64]())
        ref pred_v = self._pred.setdefault(v, Dict[Self.N, Float64]())
        pred_v[u] = weight

        _ = self._node_attr.setdefault(u, Dict[String, AttrValue]())
        _ = self._node_attr.setdefault(v, Dict[String, AttrValue]())
        ref u_edges = self._edge_attr.setdefault(u, Dict[Self.N, Dict[String, AttrValue]]())
        ref u_map = u_edges.setdefault(v, Dict[String, AttrValue]())
        u_map["weight"] = AttrValue(weight)

    def add_edges_from(mut self, edges: List[Tuple[Self.N, Self.N]]):
        for e in edges:
            self.add_edge(e[0], e[1])

    def copy(ref self, out g: DiGraph[Self.N]) raises:
        g = DiGraph[Self.N]()

        for node in self._succ.keys():
            g.add_node(node)

        for entry in self._succ.items():
            for nbr_entry in entry.value.items():
                g.add_edge(entry.key, nbr_entry.key, nbr_entry.value)

        for entry in self._graph_attr.items():
            g.set_graph_attr(entry.key, entry.value)

        for entry in self._node_attr.items():
            for kv in entry.value.items():
                g.set_node_attr(entry.key, kv.key, kv.value)

        for u_entry in self._edge_attr.items():
            for v_entry in u_entry.value.items():
                for kv in v_entry.value.items():
                    if kv.key == "weight":
                        continue
                    var tmp = kv.value
                    g.set_edge_attr(u_entry.key, v_entry.key, kv.key, tmp)

        return

    def subgraph(ref self, nodes: List[Self.N], out sg: DiGraph[Self.N]) raises:
        sg = DiGraph[Self.N]()
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

        for entry in self._succ.items():
            if not (entry.key in node_set):
                continue
            for nbr_entry in entry.value.items():
                var v: Self.N = nbr_entry.key
                if not (v in node_set):
                    continue
                sg.add_edge(entry.key, v, nbr_entry.value)

        for u_entry in self._edge_attr.items():
            if not (u_entry.key in node_set):
                continue
            for v_entry in u_entry.value.items():
                if not (v_entry.key in node_set):
                    continue
                for kv in v_entry.value.items():
                    if kv.key == "weight":
                        continue
                    var tmp = kv.value
                    sg.set_edge_attr(u_entry.key, v_entry.key, kv.key, tmp)

        return

    def clear(mut self):
        self._succ = Dict[Self.N, Dict[Self.N, Float64]]()
        self._pred = Dict[Self.N, Dict[Self.N, Float64]]()
        self._graph_attr = Dict[String, AttrValue]()
        self._node_attr = Dict[Self.N, Dict[String, AttrValue]]()
        self._edge_attr = Dict[Self.N, Dict[Self.N, Dict[String, AttrValue]]]()

    def set_graph_attr(mut self, key: String, value: AttrValue):
        self._graph_attr[key] = value

    def get_graph_attr(ref self, key: String) raises -> AttrValue:
        return self._graph_attr[key]

    def set_node_attr(mut self, node: Self.N, key: String, value: AttrValue) raises:
        if not self.has_node(node):
            raise Error("node not in graph")
        ref attrs = self._node_attr.setdefault(node, Dict[String, AttrValue]())
        attrs[key] = value

    def get_node_attr(ref self, node: Self.N, key: String) raises -> AttrValue:
        if not self.has_node(node):
            raise Error("node not in graph")
        return self._node_attr[node][key]

    def set_edge_attr(mut self, u: Self.N, v: Self.N, key: String, mut value: AttrValue) raises:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")

        if key == "weight":
            if not value.isa[Float64]():
                raise Error("weight must be Float64")
            self.add_edge(u, v, value[Float64])
            return

        ref u_edges = self._edge_attr.setdefault(u, Dict[Self.N, Dict[String, AttrValue]]())
        ref u_map = u_edges.setdefault(v, Dict[String, AttrValue]())
        u_map[key] = value

    def get_edge_attr(ref self, u: Self.N, v: Self.N, key: String) raises -> AttrValue:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")
        if key == "weight":
            return AttrValue(self._succ[u][v])
        return self._edge_attr[u][v][key]

    def out_degree(self, node: Self.N) raises -> Int:
        return len(self._succ[node])

    def in_degree(self, node: Self.N) raises -> Int:
        return len(self._pred[node])

    def degree(self, node: Self.N) raises -> Int:
        return self.in_degree(node) + self.out_degree(node)

    def has_edge(self, u: Self.N, v: Self.N) -> Bool:
        try:
            return v in self._succ[u]
        except:
            return False

    def edges(self) -> List[Tuple[Self.N, Self.N]]:
        var result = List[Tuple[Self.N, Self.N]]()
        for entry in self._succ.items():
            for nbr in entry.value.keys():
                result.append((entry.key, nbr))
        return result^

    def for_each_edge[callback: def(Self.N, Self.N) thin -> None](ref self) -> Int:
        var count = 0
        for entry in self._succ.items():
            for nbr in entry.value.keys():
                callback(entry.key, nbr)
                count += 1
        return count

    def remove_edge(mut self, u: Self.N, v: Self.N) raises:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")

        _ = self._succ[u].pop(v)
        _ = self._pred[v].pop(u)

        try:
            _ = self._edge_attr[u].pop(v)
        except:
            pass

    def remove_edges_from(mut self, edges: List[Tuple[Self.N, Self.N]]):
        for e in edges:
            if self.has_edge(e[0], e[1]):
                try:
                    self.remove_edge(e[0], e[1])
                except:
                    pass

    def remove_node(mut self, node: Self.N) raises:
        var out_neighbors = self._succ.pop(node)
        for v in out_neighbors.keys():
            _ = self._pred[v].pop(node)

            try:
                _ = self._edge_attr[node].pop(v)
            except:
                pass

        var in_neighbors = self._pred.pop(node)
        for u in in_neighbors.keys():
            _ = self._succ[u].pop(node)

            try:
                _ = self._edge_attr[u].pop(node)
            except:
                pass

        try:
            _ = self._node_attr.pop(node)
        except:
            pass

        try:
            _ = self._edge_attr.pop(node)
        except:
            pass

    def remove_nodes_from(mut self, nodes: List[Self.N]):
        for n in nodes:
            if self.has_node(n):
                try:
                    self.remove_node(n)
                except:
                    pass
