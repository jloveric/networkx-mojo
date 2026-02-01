from builtin.value import ImplicitlyCopyable
from collections import Dict, List, Set
from collections.dict import KeyElement


struct DiGraph[N: KeyElement & ImplicitlyCopyable]:
    var _succ: Dict[Self.N, Set[Self.N]]
    var _pred: Dict[Self.N, Set[Self.N]]

    fn __init__(out self):
        self._succ = Dict[Self.N, Set[Self.N]]()
        self._pred = Dict[Self.N, Set[Self.N]]()

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
