from builtin.value import ImplicitlyCopyable
from collections import Dict, List, Set
from collections.dict import KeyElement
from utils import Variant

comptime AttrValue = Variant[Int, Float64, Bool, String]


struct MultiGraph[N: KeyElement & ImplicitlyCopyable]:
    var _adj: Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]]
    var _graph_attr: Dict[String, AttrValue]
    var _node_attr: Dict[Self.N, Dict[String, AttrValue]]
    var _edge_attr: Dict[Self.N, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]]

    fn __init__(out self):
        self._adj = Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]]()
        self._graph_attr = Dict[String, AttrValue]()
        self._node_attr = Dict[Self.N, Dict[String, AttrValue]]()
        self._edge_attr = Dict[Self.N, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]]()

    fn __len__(self) -> Int:
        return self.number_of_nodes()

    fn __contains__(self, node: Self.N) -> Bool:
        return self.has_node(node)

    fn __iter__(self) -> Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]].IteratorType[iterable_mut=False, iterable_origin=origin_of(self._adj)]:
        return self._adj.keys()

    fn number_of_nodes(self) -> Int:
        return len(self._adj)

    fn order(self) -> Int:
        return self.number_of_nodes()

    fn number_of_edges(self) -> Int:
        var total = 0
        var processed = Set[Self.N]()
        for entry in self._adj.items():
            for nbr_entry in entry.value.items():
                var v = nbr_entry.key
                if v in processed:
                    continue
                total += len(nbr_entry.value)
            processed.add(entry.key)
        return total

    fn size(self) -> Int:
        return self.number_of_edges()

    fn is_directed(self) -> Bool:
        return False

    fn is_multigraph(self) -> Bool:
        return True

    fn has_node(self, node: Self.N) -> Bool:
        return node in self._adj

    fn nodes(ref self) -> List[Self.N]:
        var result = List[Self.N]()
        for node in self._adj.keys():
            result.append(node)
        return result^

    fn adj_view(ref self) -> ref[self._adj] Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]]:
        return self._adj

    fn adj(ref self) -> ref[self._adj] Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]]:
        return self._adj

    fn __getitem__(ref self, node: Self.N) raises -> Dict[Self.N, Dict[Int, Float64]]:
        return self._adj[node].copy()

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
        _ = self._adj.setdefault(node, Dict[Self.N, Dict[Int, Float64]]())
        _ = self._node_attr.setdefault(node, Dict[String, AttrValue]())
        _ = self._edge_attr.setdefault(node, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]())

    fn add_nodes_from(mut self, nodes: List[Self.N]):
        for node in nodes:
            self.add_node(node)

    fn _next_key(ref self, u: Self.N, v: Self.N) raises -> Int:
        try:
            ref nbrs = self._adj[u]
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

        ref nbrs_u = self._adj.setdefault(u, Dict[Self.N, Dict[Int, Float64]]())
        ref nbrs_v = self._adj.setdefault(v, Dict[Self.N, Dict[Int, Float64]]())

        ref map_u = nbrs_u.setdefault(v, Dict[Int, Float64]())
        map_u[k] = weight
        if u != v:
            ref map_v = nbrs_v.setdefault(u, Dict[Int, Float64]())
            map_v[k] = weight

        _ = self._node_attr.setdefault(u, Dict[String, AttrValue]())
        _ = self._node_attr.setdefault(v, Dict[String, AttrValue]())

        ref u_edges = self._edge_attr.setdefault(u, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]())
        ref v_edges = self._edge_attr.setdefault(v, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]())

        ref u_v = u_edges.setdefault(v, Dict[Int, Dict[String, AttrValue]]())
        ref u_map = u_v.setdefault(k, Dict[String, AttrValue]())
        u_map["weight"] = AttrValue(weight)

        if u != v:
            ref v_u = v_edges.setdefault(u, Dict[Int, Dict[String, AttrValue]]())
            ref v_map = v_u.setdefault(k, Dict[String, AttrValue]())
            v_map["weight"] = AttrValue(weight)

        return k

    fn add_edges_from(mut self, edges: List[Tuple[Self.N, Self.N]]) raises:
        for e in edges:
            _ = self.add_edge(e[0], e[1])

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

    fn bfs_tree(ref self, source: Self.N, out t: MultiGraph[Self.N]) raises:
        if not self.has_node(source):
            raise Error("node not in graph")

        t = MultiGraph[Self.N]()
        t.add_node(source)

        var seen = Set[Self.N]()
        seen.add(source)

        var queue = List[Self.N]()
        queue.append(source)
        var head = 0

        while head < len(queue):
            var u = queue[head]
            head += 1
            for nbr_entry in self._adj[u].items():
                var v = nbr_entry.key
                if v in seen:
                    continue
                seen.add(v)
                queue.append(v)
                var w = 1.0
                for k_entry in nbr_entry.value.items():
                    w = k_entry.value
                    break
                _ = t.add_edge(u, v, w)

        return

    fn dfs_tree(ref self, source: Self.N, out t: MultiGraph[Self.N]) raises:
        if not self.has_node(source):
            raise Error("node not in graph")

        t = MultiGraph[Self.N]()
        t.add_node(source)

        var seen = Set[Self.N]()
        seen.add(source)

        var stack = List[Self.N]()
        stack.append(source)

        while len(stack) > 0:
            var u = stack.pop()
            for nbr_entry in self._adj[u].items():
                var v = nbr_entry.key
                if v in seen:
                    continue
                seen.add(v)
                stack.append(v)
                var w = 1.0
                for k_entry in nbr_entry.value.items():
                    w = k_entry.value
                    break
                _ = t.add_edge(u, v, w)

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

    fn copy(ref self, out g: MultiGraph[Self.N]) raises:
        g = MultiGraph[Self.N]()

        for node in self._adj.keys():
            g.add_node(node)

        var processed = Set[Self.N]()
        for entry in self._adj.items():
            for nbr_entry in entry.value.items():
                var v = nbr_entry.key
                if v in processed:
                    continue
                for k_entry in nbr_entry.value.items():
                    _ = g.add_edge(entry.key, v, k_entry.value, k_entry.key)
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
                for k_entry in v_entry.value.items():
                    for kv in k_entry.value.items():
                        if kv.key == "weight":
                            continue
                        var tmp = kv.value
                        g.set_edge_attr(u_entry.key, v, k_entry.key, kv.key, tmp)
            processed_edges.add(u_entry.key)

        return

    fn subgraph(ref self, nodes: List[Self.N], out sg: MultiGraph[Self.N]) raises:
        sg = MultiGraph[Self.N]()
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
                for k_entry in nbr_entry.value.items():
                    _ = sg.add_edge(entry.key, v, k_entry.value, k_entry.key)
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
                for k_entry in v_entry.value.items():
                    for kv in k_entry.value.items():
                        if kv.key == "weight":
                            continue
                        var tmp = kv.value
                        sg.set_edge_attr(u_entry.key, v, k_entry.key, kv.key, tmp)
            processed_edges.add(u_entry.key)

        return

    fn has_edge(self, u: Self.N, v: Self.N) -> Bool:
        try:
            return v in self._adj[u]
        except:
            return False

    fn has_edge_key(self, u: Self.N, v: Self.N, key: Int) -> Bool:
        try:
            return key in self._adj[u][v]
        except:
            return False

    fn edges(self) -> List[Tuple[Self.N, Self.N, Int]]:
        var result = List[Tuple[Self.N, Self.N, Int]]()
        var processed = Set[Self.N]()
        for entry in self._adj.items():
            for nbr_entry in entry.value.items():
                var v = nbr_entry.key
                if v in processed:
                    continue
                for k in nbr_entry.value.keys():
                    result.append((entry.key, v, k))
            processed.add(entry.key)
        return result^

    fn for_each_edge[callback: fn(Self.N, Self.N, Int)](ref self) -> Int:
        var count = 0
        var processed = Set[Self.N]()
        for entry in self._adj.items():
            for nbr_entry in entry.value.items():
                var v = nbr_entry.key
                if v in processed:
                    continue
                for k in nbr_entry.value.keys():
                    callback(entry.key, v, k)
                    count += 1
            processed.add(entry.key)
        return count

    fn degree(self, node: Self.N) raises -> Int:
        var d = 0
        for entry in self._adj[node].items():
            d += len(entry.value)
        if node in self._adj[node]:
            d += len(self._adj[node][node])
        return d

    fn clear(mut self):
        self._adj = Dict[Self.N, Dict[Self.N, Dict[Int, Float64]]]()
        self._graph_attr = Dict[String, AttrValue]()
        self._node_attr = Dict[Self.N, Dict[String, AttrValue]]()
        self._edge_attr = Dict[Self.N, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]]()

    fn remove_node(mut self, node: Self.N) raises:
        var neighbors = self._adj.pop(node)
        for entry in neighbors.items():
            var nbr = entry.key
            if nbr == node:
                continue
            try:
                _ = self._adj[nbr].pop(node)
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

        for entry in neighbors.items():
            var nbr = entry.key
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

    fn remove_edge(mut self, u: Self.N, v: Self.N, key: Int) raises:
        if not self.has_edge_key(u, v, key):
            raise Error("edge not in graph")

        ref u_adj = self._adj[u]
        ref keys_uv = u_adj[v]

        try:
            _ = keys_uv.pop(key)
        except:
            raise Error("edge not in graph")

        if len(keys_uv) == 0:
            try:
                _ = u_adj.pop(v)
            except:
                pass

        if u != v:
            ref v_adj = self._adj[v]
            ref keys_vu = v_adj[u]

            try:
                _ = keys_vu.pop(key)
            except:
                raise Error("edge not in graph")

            if len(keys_vu) == 0:
                try:
                    _ = v_adj.pop(u)
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

        if u != v:
            try:
                ref v_edges = self._edge_attr[v]
                try:
                    ref v_u = v_edges[u]
                    try:
                        _ = v_u.pop(key)
                    except:
                        pass
                except:
                    pass
            except:
                pass

    fn remove_edges_from(mut self, edges: List[Tuple[Self.N, Self.N, Int]]):
        for e in edges:
            if self.has_edge_key(e[0], e[1], e[2]):
                try:
                    self.remove_edge(e[0], e[1], e[2])
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
        ref v_edges = self._edge_attr.setdefault(v, Dict[Self.N, Dict[Int, Dict[String, AttrValue]]]())

        ref u_v = u_edges.setdefault(v, Dict[Int, Dict[String, AttrValue]]())
        ref u_map = u_v.setdefault(edge_key, Dict[String, AttrValue]())
        u_map[key] = value

        if u != v:
            ref v_u = v_edges.setdefault(u, Dict[Int, Dict[String, AttrValue]]())
            ref v_map = v_u.setdefault(edge_key, Dict[String, AttrValue]())
            v_map[key] = value

    fn get_edge_attr(ref self, u: Self.N, v: Self.N, edge_key: Int, key: String) raises -> AttrValue:
        if not self.has_edge_key(u, v, edge_key):
            raise Error("edge not in graph")
        if key == "weight":
            return AttrValue(self._adj[u][v][edge_key])
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
