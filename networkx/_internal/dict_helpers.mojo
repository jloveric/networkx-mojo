from std.collections import Dict, List
from std.collections.dict import KeyElement
from std.traits import Deinitable
from std.traits.copyable import ImplicitlyCopyable

comptime _NodeConstraint = KeyElement & Deinitable & ImplicitlyCopyable


def dict_get_nn[N: _NodeConstraint](d: Dict[N, N], key: N) raises -> N:
    return d[key]


def dict_get_nf[N: _NodeConstraint](d: Dict[N, Float64], key: N) raises -> Float64:
    return d[key]


def dict_get_ni[N: _NodeConstraint](d: Dict[N, Int], key: N) raises -> Int:
    return d[key]


def dict_contains_key[N: _NodeConstraint, V: Deinitable](d: Dict[N, V], key: N) -> Bool:
    return key in d


def dict_contains_nested_key[N: _NodeConstraint, V: Copyable & Deinitable](d: Dict[N, Dict[N, V]], outer: N, inner: N) raises -> Bool:
    return inner in d[outer]


def dict_remove_nested_key[N: _NodeConstraint, V: Copyable & Deinitable](mut d: Dict[N, Dict[N, V]], outer: N, inner: N) raises:
    _ = d[outer].pop(inner)


def dict_neighbor_keys[N: _NodeConstraint, V: Copyable & Deinitable](ref adj: Dict[N, Dict[N, V]], u: N) raises -> List[N]:
    var out = List[N]()
    for entry in adj[u].items():
        out.append(entry.key)
    return out^


def dict_adj_entries[N: _NodeConstraint, W: Copyable & Deinitable & ImplicitlyCopyable](ref adj: Dict[N, Dict[N, W]], u: N) raises -> List[Tuple[N, W]]:
    var out = List[Tuple[N, W]]()
    for e in adj[u].items():
        out.append((e.key, e.value))
    return out^


def dict_set_inner_map[N: _NodeConstraint, V: Copyable & Deinitable](mut outer: Dict[N, Dict[N, V]], key: N, var inner: Dict[N, V]):
    outer[key] = inner^


def dict_get_nf_cell[N: _NodeConstraint](ref dist: Dict[N, Dict[N, Float64]], u: N, v: N) raises -> Float64:
    return dist[u][v]


def dict_set_nf_cell[N: _NodeConstraint](mut dist: Dict[N, Dict[N, Float64]], u: N, v: N, w: Float64) raises:
    dist[u][v] = w


def dict_get_nn_cell[N: _NodeConstraint](ref pred: Dict[N, Dict[N, N]], u: N, v: N) raises -> N:
    return pred[u][v]


def dict_set_nn_cell[N: _NodeConstraint](mut pred: Dict[N, Dict[N, N]], u: N, v: N, w: N) raises:
    pred[u][v] = w


