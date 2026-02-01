from builtin.value import ImplicitlyCopyable
from collections import Dict, List
from collections.dict import KeyElement
from utils import Variant

comptime AttrValue = Variant[Int, Float64, Bool, String]


struct MultiDiGraph[N: KeyElement & ImplicitlyCopyable]:
    var _succ: Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]]
    var _pred: Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]]
    var _graph_attr: Dict[String, AttrValue]
    var _node_attr: Dict[Self.N, Dict[String, AttrValue]]
    var _edge_attr: Dict[Self.N, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]]

    fn __init__(out self):
        self._succ = Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]]()
        self._pred = Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]]()
        self._graph_attr = Dict[String, AttrValue]()
        self._node_attr = Dict[Self.N, Dict[String, AttrValue]]()
        self._edge_attr = Dict[Self.N, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]]()

    fn __len__(self) -> Int:
        return self.number_of_nodes()

    fn __contains__(self, node: Self.N) -> Bool:
        return self.has_node(node)

    fn __iter__(self) -> Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]].IteratorType[iterable_mut=False, iterable_origin=origin_of(self._succ)]:
        return self._succ.keys()

    fn number_of_nodes(self) -> Int:
        return len(self._succ)

    fn order(self) -> Int:
        return self.number_of_nodes()

    fn number_of_edges(self) -> Int:
        var total = 0
        for entry in self._succ.items():
            for nbr_entry in entry.value.items():
                total += len(nbr_entry.value)
        return total

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

    fn add_node(mut self, node: Self.N):
        _ = self._succ.setdefault(node, Dict[Self.N, Dict[Int, Float64]]())
        _ = self._pred.setdefault(node, Dict[Self.N, Dict[Int, Float64]]())
        _ = self._node_attr.setdefault(node, Dict[String, AttrValue]())
        _ = self._edge_attr.setdefault(node, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]())

    fn add_nodes_from(mut self, nodes: List[Self.N]):
        for node in nodes:
            self.add_node(node)

    fn _next_key(ref self, u: Self.N, v: Self.N) raises -> Int:
        try:
            ref nbrs = self._succ[u]
            try:
                ref keys = nbrs[v]
                return len(keys)
            except:
                return 0
        except:
            return 0

    fn add_edge(mut self, u: Self.N, v: Self.N, weight: Float64 = 1.0, key: Int = -1) raises -> Int:
        self.add_node(u)
        self.add_node(v)

        var k = key
        if k < 0:
            k = self._next_key(u, v)

        ref succ_u = self._succ.setdefault(u, Dict[Self.N, Dict[Int, Float64]]())
        ref pred_v = self._pred.setdefault(v, Dict[Self.N, Dict[Int, Float64]]())
        _ = self._succ.setdefault(v, Dict[Self.N, Dict[Int, Float64]]())
        _ = self._pred.setdefault(u, Dict[Self.N, Dict[Int, Float64]]())

        ref map_u = succ_u.setdefault(v, Dict[Int, Float64]())
        map_u[k] = weight
        ref map_p = pred_v.setdefault(u, Dict[Int, Float64]())
        map_p[k] = weight

        _ = self._node_attr.setdefault(u, Dict[String, AttrValue]())
        _ = self._node_attr.setdefault(v, Dict[String, AttrValue]())

        ref u_edges = self._edge_attr.setdefault(u, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]())
        ref u_v = u_edges.setdefault(v, Dict[Int, Dict[String, AttrValue]]())
        ref u_map_attr = u_v.setdefault(k, Dict[String, AttrValue]())
        u_map_attr["weight"] = AttrValue(weight)

        return k

    fn has_edge(self, u: Self.N, v: Self.N) -> Bool:
        try:
            return v in self._succ[u]
        except:
            return False

    fn has_edge_key(self, u: Self.N, v: Self.N, key: Int) -> Bool:
        try:
            return key in self._succ[u][v]
        except:
            return False

    fn edges(self) -> List[Tuple[Self.N, Self.N, Int]]:
        var result = List[Tuple[Self.N, Self.N, Int]]()
        for entry in self._succ.items():
            for nbr_entry in entry.value.items():
                for k in nbr_entry.value.keys():
                    result.append((entry.key, nbr_entry.key, k))
        return result^

    fn for_each_edge[callback: fn(Self.N, Self.N, Int)](ref self) -> Int:
        var count = 0
        for entry in self._succ.items():
            for nbr_entry in entry.value.items():
                for k in nbr_entry.value.keys():
                    callback(entry.key, nbr_entry.key, k)
                    count += 1
        return count

    fn out_degree(self, node: Self.N) raises -> Int:
        var d = 0
        for entry in self._succ[node].items():
            d += len(entry.value)
        return d

    fn in_degree(self, node: Self.N) raises -> Int:
        var d = 0
        for entry in self._pred[node].items():
            d += len(entry.value)
        return d

    fn remove_edge(mut self, u: Self.N, v: Self.N, key: Int) raises:
        if not self.has_edge_key(u, v, key):
            raise Error("edge not in graph")

        ref u_succ = self._succ[u]
        ref keys_uv = u_succ[v]

        try:
            _ = keys_uv.pop(key)
        except:
            raise Error("edge not in graph")

        if len(keys_uv) == 0:
            try:
                _ = u_succ.pop(v)
            except:
                pass

        ref v_pred = self._pred[v]
        ref keys_vu = v_pred[u]

        try:
            _ = keys_vu.pop(key)
        except:
            raise Error("edge not in graph")

        if len(keys_vu) == 0:
            try:
                _ = v_pred.pop(u)
            except:
                pass

        try:
            ref u_edges = self._edge_attr[u]
            try:
                ref u_v = u_edges[v]
                try:
                    _ = u_v.pop(key)
                except:
                    pass
            except:
                pass
        except:
            pass

    fn set_edge_attr(mut self, u: Self.N, v: Self.N, edge_key: Int, key: String, mut value: AttrValue) raises:
        if not self.has_edge_key(u, v, edge_key):
            raise Error("edge not in graph")

        if key == "weight":
            if not value.isa[Float64]():
                raise Error("weight must be Float64")
            _ = self.add_edge(u, v, value[Float64], edge_key)
            return

        ref u_edges = self._edge_attr.setdefault(u, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]())
        ref u_v = u_edges.setdefault(v, Dict[Int, Dict[String, AttrValue]]())
        ref u_map = u_v.setdefault(edge_key, Dict[String, AttrValue]())
        u_map[key] = value

    fn get_edge_attr(ref self, u: Self.N, v: Self.N, edge_key: Int, key: String) raises -> AttrValue:
        if not self.has_edge_key(u, v, edge_key):
            raise Error("edge not in graph")
        if key == "weight":
            return AttrValue(self._succ[u][v][edge_key])
        return self._edge_attr[u][v][edge_key][key]

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
