from builtin.value import ImplicitlyCopyable
from collections import Dict, List, Set
from collections.dict import KeyElement

struct Graph[N: KeyElement & ImplicitlyCopyable]:
    var _adj: Dict[Self.N, Set[Self.N]]

    fn __init__(out self):
        self._adj = Dict[Self.N, Set[Self.N]]()

    fn __len__(self) -> Int:
        return self.number_of_nodes()

    fn __contains__(self, node: Self.N) -> Bool:
        return self.has_node(node)

    fn __iter__(ref self) -> List[Self.N]:
        return self.nodes()

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

    fn has_node(self, node: Self.N) -> Bool:
        return node in self._adj

    fn nodes(ref self) -> List[Self.N]:
        var result = List[Self.N]()
        for node in self._adj.keys():
            result.append(node)
        return result^

    fn adj(ref self, node: Self.N) raises -> List[Self.N]:
        return self.neighbors(node)

    fn neighbors(ref self, node: Self.N) raises -> List[Self.N]:
        var result = List[Self.N]()
        for nbr in self._adj[node]:
            result.append(nbr)
        return result^

    fn add_node(mut self, node: Self.N):
        _ = self._adj.setdefault(node, Set[Self.N]())

    fn add_nodes_from(mut self, nodes: List[Self.N]):
        for node in nodes:
            self.add_node(node)

    fn add_edge(mut self, u: Self.N, v: Self.N):
        ref neighbors_u = self._adj.setdefault(u, Set[Self.N]())
        ref neighbors_v = self._adj.setdefault(v, Set[Self.N]())

        neighbors_u.add(v)
        if u != v:
            neighbors_v.add(u)

    fn add_edges_from(mut self, edges: List[Tuple[Self.N, Self.N]]):
        for e in edges:
            self.add_edge(e[0], e[1])

    fn clear(mut self):
        self._adj = Dict[Self.N, Set[Self.N]]()

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
            for nbr in entry.value:
                if nbr in processed:
                    continue
                result.append((entry.key, nbr))
            processed.add(entry.key)
        return result^

    fn remove_edge(mut self, u: Self.N, v: Self.N) raises:
        if not self.has_edge(u, v):
            raise Error("edge not in graph")

        self._adj[u].discard(v)
        if u != v:
            self._adj[v].discard(u)

    fn remove_node(mut self, node: Self.N) raises:
        var neighbors = self._adj.pop(node)
        for nbr in neighbors:
            if nbr == node:
                continue
            self._adj[nbr].discard(node)
