from builtin.value import ImplicitlyCopyable
from collections import Dict, List, Set
from collections.dict import KeyElement


struct _HeapItem[N: KeyElement & ImplicitlyCopyable](ImplicitlyCopyable):
    var prio: Float64
    var count: Int
    var node: Self.N

    fn __init__(out self, prio: Float64, count: Int, node: Self.N):
        self.prio = prio
        self.count = count
        self.node = node

struct _MinHeap[N: KeyElement & ImplicitlyCopyable]:
    var _data: List[_HeapItem[Self.N]]

    fn __init__(out self):
        self._data = List[_HeapItem[Self.N]]()

    fn is_empty(self) -> Bool:
        return len(self._data) == 0

    fn _less(self, a: _HeapItem[Self.N], b: _HeapItem[Self.N]) -> Bool:
        if a.prio < b.prio:
            return True
        if a.prio > b.prio:
            return False
        return a.count < b.count

    fn push(mut self, item: _HeapItem[Self.N]):
        self._data.append(item)
        var i = len(self._data) - 1
        while i > 0:
            var parent = (i - 1) // 2
            if not self._less(self._data[i], self._data[parent]):
                break
            var tmp = self._data[parent]
            self._data[parent] = self._data[i]
            self._data[i] = tmp
            i = parent

    fn pop_min(mut self) raises -> _HeapItem[Self.N]:
        if len(self._data) == 0:
            raise Error("empty heap")
        var result = self._data[0]
        var last = self._data.pop()
        if len(self._data) == 0:
            return result
        self._data[0] = last
        var i = 0
        while True:
            var left = 2 * i + 1
            var right = 2 * i + 2
            if left >= len(self._data):
                break
            var smallest = left
            if right < len(self._data) and self._less(self._data[right], self._data[left]):
                smallest = right
            if not self._less(self._data[smallest], self._data[i]):
                break
            var tmp = self._data[i]
            self._data[i] = self._data[smallest]
            self._data[smallest] = tmp
            i = smallest
        return result


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


struct DiGraph[N: KeyElement & ImplicitlyCopyable]:
    var _succ: Dict[Self.N, Set[Self.N]]
    var _pred: Dict[Self.N, Set[Self.N]]

    fn __init__(out self):
        self._succ = Dict[Self.N, Set[Self.N]]()
        self._pred = Dict[Self.N, Set[Self.N]]()

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

    fn __iter__(ref self) -> List[Self.N]:
        return self.nodes()

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
            for v in self._succ[u]:
                if v in visited:
                    continue
                visited.add(v)
                parents[v] = u
                if v == target:
                    return self._reconstruct_path(parents, source, target)
                queue.append(v)

        raise Error("no path")

    fn dijkstra_path(ref self, source: Self.N, target: Self.N) raises -> List[Self.N]:
        return self.dijkstra_path_weighted[_unit_weight](source, target)

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
            for v in self._succ[u]:
                if v in finalized:
                    continue
                var nd = du + weight_fn(u, v)
                var better = False
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
        return self.astar_path_weighted[_unit_weight, _zero_heuristic](source, target)

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
            for v in self._succ[u]:
                if v in closed:
                    continue
                var tentative = gu + weight_fn(u, v)
                var better = False
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

    fn successors(ref self, node: Self.N) raises -> List[Self.N]:
        var result = List[Self.N]()
        for nbr in self._succ[node]:
            result.append(nbr)
        return result^

    fn neighbors(ref self, node: Self.N) raises -> List[Self.N]:
        return self.successors(node)

    fn adj(ref self, node: Self.N) raises -> List[Self.N]:
        return self.successors(node)

    fn predecessors(ref self, node: Self.N) raises -> List[Self.N]:
        var result = List[Self.N]()
        for nbr in self._pred[node]:
            result.append(nbr)
        return result^

    fn add_node(mut self, node: Self.N):
        _ = self._succ.setdefault(node, Set[Self.N]())
        _ = self._pred.setdefault(node, Set[Self.N]())

    fn add_nodes_from(mut self, nodes: List[Self.N]):
        for node in nodes:
            self.add_node(node)

    fn add_edge(mut self, u: Self.N, v: Self.N):
        ref succ_u = self._succ.setdefault(u, Set[Self.N]())
        _ = self._pred.setdefault(u, Set[Self.N]())

        _ = self._succ.setdefault(v, Set[Self.N]())
        ref pred_v = self._pred.setdefault(v, Set[Self.N]())

        succ_u.add(v)
        pred_v.add(u)

    fn add_edges_from(mut self, edges: List[Tuple[Self.N, Self.N]]):
        for e in edges:
            self.add_edge(e[0], e[1])

    fn clear(mut self):
        self._succ = Dict[Self.N, Set[Self.N]]()
        self._pred = Dict[Self.N, Set[Self.N]]()

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
            for nbr in entry.value:
                result.append((entry.key, nbr))
        return result^

    fn remove_edge(mut self, u: Self.N, v: Self.N) raises:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")

        self._succ[u].discard(v)
        self._pred[v].discard(u)

    fn remove_node(mut self, node: Self.N) raises:
        var out_neighbors = self._succ.pop(node)
        for v in out_neighbors:
            self._pred[v].discard(node)

        var in_neighbors = self._pred.pop(node)
        for u in in_neighbors:
            self._succ[u].discard(node)
