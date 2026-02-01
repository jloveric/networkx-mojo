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

comptime AttrValue = Variant[Int, Float64, Bool, String]

struct DiGraph[N: KeyElement & ImplicitlyCopyable]:
    var _succ: Dict[Self.N, Dict[Self.N, Float64]]
    var _pred: Dict[Self.N, Dict[Self.N, Float64]]
    var _graph_attr: Dict[String, AttrValue]
    var _node_attr: Dict[Self.N, Dict[String, AttrValue]]
    var _edge_attr: Dict[Self.N, Dict[Self.N, Dict[String, AttrValue]]]

    fn __init__(out self):
        self._succ = Dict[Self.N, Dict[Self.N, Float64]]()
        self._pred = Dict[Self.N, Dict[Self.N, Float64]]()
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

    fn __iter__(self) -> Dict[Self.N, Dict[Self.N, Float64]].IteratorType[iterable_mut=False, iterable_origin=origin_of(self._succ)]:
        return self._succ.keys()

    fn number_of_nodes(self) -> Int:
        return len(self._succ)

    fn order(self) -> Int:
        return self.number_of_nodes()

    fn number_of_edges(self) -> Int:
        var total_out_degree = 0
        for e in self._succ.items():
            total_out_degree += len(e.value)
        return total_out_degree

    fn size(self) -> Int:
        return self.number_of_edges()

    fn is_directed(self) -> Bool:
        return True

    fn topological_sort(ref self) raises -> List[Self.N]:
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
            var u = queue[head]
            head += 1
            out.append(u)

            for v in self._succ[u].keys():
                indeg[v] = indeg[v] - 1
                if indeg[v] == 0:
                    queue.append(v)

        if len(out) != len(indeg):
            raise Error("graph has a cycle")
        return out^

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
            for v in self._succ[u].keys():
                if v in visited:
                    continue
                visited.add(v)
                parents[v] = u
                if v == target:
                    return self._reconstruct_path(parents, source, target)
                queue.append(v)

        raise Error("no path")

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
            for e in self._succ[u].items():
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
            for v in self._succ[u].keys():
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
            for e in self._succ[u].items():
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
            for v in self._succ[u].keys():
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
        return node in self._succ

    fn nodes(ref self) -> List[Self.N]:
        var result = List[Self.N]()
        for node in self._succ.keys():
            result.append(node)
        return result^

    fn succ_view(ref self) -> ref[self._succ] Dict[Self.N, Dict[Self.N, Float64]]:
        return self._succ

    fn pred_view(ref self) -> ref[self._pred] Dict[Self.N, Dict[Self.N, Float64]]:
        return self._pred

    fn for_each_node[callback: fn(Self.N)](ref self) -> Int:
        var count = 0
        for node in self._succ.keys():
            callback(node)
            count += 1
        return count

    fn successors(ref self, node: Self.N) raises -> List[Self.N]:
        var result = List[Self.N]()
        for nbr in self._succ[node].keys():
            result.append(nbr)
        return result^

    fn for_each_successor[callback: fn(Self.N)](ref self, node: Self.N) raises -> Int:
        var count = 0
        for nbr in self._succ[node].keys():
            callback(nbr)
            count += 1
        return count

    fn neighbors(ref self, node: Self.N) raises -> List[Self.N]:
        return self.successors(node)

    fn adj(ref self, node: Self.N) raises -> List[Self.N]:
        return self.successors(node)

    fn predecessors(ref self, node: Self.N) raises -> List[Self.N]:
        var result = List[Self.N]()
        for nbr in self._pred[node].keys():
            result.append(nbr)
        return result^

    fn for_each_predecessor[callback: fn(Self.N)](ref self, node: Self.N) raises -> Int:
        var count = 0
        for nbr in self._pred[node].keys():
            callback(nbr)
            count += 1
        return count

    fn add_node(mut self, node: Self.N):
        _ = self._succ.setdefault(node, Dict[Self.N, Float64]())
        _ = self._pred.setdefault(node, Dict[Self.N, Float64]())
        _ = self._node_attr.setdefault(node, Dict[String, AttrValue]())
        _ = self._edge_attr.setdefault(node, Dict[Self.N, Dict[String, AttrValue]]())

    fn add_nodes_from(mut self, nodes: List[Self.N]):
        for node in nodes:
            self.add_node(node)

    fn add_edge(mut self, u: Self.N, v: Self.N, weight: Float64 = 1.0):
        ref succ_u = self._succ.setdefault(u, Dict[Self.N, Float64]())
        _ = self._pred.setdefault(u, Dict[Self.N, Float64]())

        _ = self._succ.setdefault(v, Dict[Self.N, Float64]())
        ref pred_v = self._pred.setdefault(v, Dict[Self.N, Float64]())

        succ_u[v] = weight
        pred_v[u] = weight

        _ = self._node_attr.setdefault(u, Dict[String, AttrValue]())
        _ = self._node_attr.setdefault(v, Dict[String, AttrValue]())
        ref u_edges = self._edge_attr.setdefault(u, Dict[Self.N, Dict[String, AttrValue]]())
        ref u_map = u_edges.setdefault(v, Dict[String, AttrValue]())
        u_map["weight"] = AttrValue(weight)

    fn add_edges_from(mut self, edges: List[Tuple[Self.N, Self.N]]):
        for e in edges:
            self.add_edge(e[0], e[1])

    fn clear(mut self):
        self._succ = Dict[Self.N, Dict[Self.N, Float64]]()
        self._pred = Dict[Self.N, Dict[Self.N, Float64]]()
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
        ref u_map = u_edges.setdefault(v, Dict[String, AttrValue]())
        u_map[key] = value

    fn get_edge_attr(ref self, u: Self.N, v: Self.N, key: String) raises -> AttrValue:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")
        if key == "weight":
            return AttrValue(self._succ[u][v])
        return self._edge_attr[u][v][key]

    fn out_degree(self, node: Self.N) raises -> Int:
        return len(self._succ[node])

    fn in_degree(self, node: Self.N) raises -> Int:
        return len(self._pred[node])

    fn degree(self, node: Self.N) raises -> Int:
        return self.in_degree(node) + self.out_degree(node)

    fn has_edge(self, u: Self.N, v: Self.N) -> Bool:
        try:
            return v in self._succ[u]
        except:
            return False

    fn edges(self) -> List[Tuple[Self.N, Self.N]]:
        var result = List[Tuple[Self.N, Self.N]]()
        for entry in self._succ.items():
            for nbr in entry.value.keys():
                result.append((entry.key, nbr))
        return result^

    fn for_each_edge[callback: fn(Self.N, Self.N)](ref self) -> Int:
        var count = 0
        for entry in self._succ.items():
            for nbr in entry.value.keys():
                callback(entry.key, nbr)
                count += 1
        return count

    fn remove_edge(mut self, u: Self.N, v: Self.N) raises:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")

        _ = self._succ[u].pop(v)
        _ = self._pred[v].pop(u)

        try:
            _ = self._edge_attr[u].pop(v)
        except:
            pass

    fn remove_node(mut self, node: Self.N) raises:
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
