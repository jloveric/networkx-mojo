# NetworkX-Mojo API Audit (vs Python NetworkX)

This repository is a Mojo reimplementation of a small, performance-oriented subset of Python NetworkX.

## Scope and notes

- This audit focuses on **API shape parity**, not algorithmic completeness.
- The current implementation is intentionally **Mojo-first**:
  - Many APIs return `List[...]` rather than NetworkX "views" (`NodeView`, `EdgeView`, `AdjacencyView`).
  - Some constructors / builders use Mojo `out`-parameter patterns internally, but are callable ergonomically (e.g. `var h = g.copy()`).
  - Attributes are stored as `AttrValue = Variant[Int, Float64, Bool, String]`.
  - Edge weight is treated as a special attribute name: `"weight"`.

## Tier definitions

- **Tier 0 (core containers)**: must-have methods/properties for typical graph construction, mutation, inspection.
- **Tier 1 (common algorithms / quality-of-life)**: frequently used algorithms and helpers.
- **Tier 2 (extended)**: advanced algorithms, rich view/dict compatibility, serialization, etc.

---

# Implemented APIs (current)

## `Graph`

Implemented:
- Construction / basics:
  - `__len__`, `__contains__`, `__iter__`
  - `number_of_nodes`, `order`, `number_of_edges`, `size`
  - `is_directed`
- Core structure:
  - `add_node`, `add_nodes_from`
  - `add_edge`, `add_edges_from`
  - `remove_node`, `remove_edge`
  - `has_node`, `has_edge`
  - `nodes` (returns `List[N]`)
  - `neighbors`, `adj` (returns `List[N]`)
  - `adj_view` (returns `ref` to adjacency dict)
  - `degree`
  - `edges` (returns `List[(u,v)]`)
  - `clear`
  - `copy`, `subgraph`
- Traversal helpers:
  - `for_each_node`, `for_each_neighbor`, `for_each_edge`
- Attributes:
  - `set_graph_attr`, `get_graph_attr`
  - `set_node_attr`, `get_node_attr`
  - `set_edge_attr`, `get_edge_attr`
- Algorithms:
  - `connected_components`, `number_connected_components`, `is_connected`
  - `minimum_spanning_tree`
  - `shortest_path` (unweighted)
  - `dijkstra_path`, `dijkstra_path_weighted`
  - `astar_path`, `astar_path_weighted`

## `DiGraph`

Implemented:
- Construction / basics:
  - `__len__`, `__contains__`, `__iter__`
  - `number_of_nodes`, `order`, `number_of_edges`, `size`
  - `is_directed`
- Core structure:
  - `add_node`, `add_nodes_from`
  - `add_edge`, `add_edges_from`
  - `remove_node`, `remove_edge`
  - `has_node`, `has_edge`
  - `nodes` (returns `List[N]`)
  - `successors`, `predecessors`, `neighbors`, `adj`
  - `succ_view`, `pred_view`
  - `out_degree`, `in_degree`, `degree`
  - `edges` (returns `List[(u,v)]`)
  - `clear`
  - `copy`, `subgraph`
- Traversal helpers:
  - `for_each_node`, `for_each_successor`, `for_each_predecessor`, `for_each_edge`
- Attributes:
  - `set_graph_attr`, `get_graph_attr`
  - `set_node_attr`, `get_node_attr`
  - `set_edge_attr`, `get_edge_attr`
- Algorithms:
  - `topological_sort`
  - `is_dag`
  - `ancestors`, `descendants`
  - `shortest_path` (unweighted)
  - `dijkstra_path`, `dijkstra_path_weighted`
  - `astar_path`, `astar_path_weighted`

## `MultiGraph`

Implemented:
- Construction / basics:
  - `__len__`, `__contains__`, `__iter__`
  - `number_of_nodes`, `order`, `number_of_edges`, `size`
  - `is_directed`
- Core structure:
  - `add_node`, `add_nodes_from`
  - `add_edge` (supports `key`), `add_edges_from`
  - `remove_node`, `remove_edge(u,v,key)`
  - `has_node`, `has_edge`, `has_edge_key`
  - `nodes`, `neighbors`, `adj`, `adj_view`
  - `degree`
  - `edges` (returns `List[(u,v,key)]`)
  - `clear`
  - `copy`, `subgraph`
- Traversal helpers:
  - `for_each_neighbor`, `for_each_edge`
- Attributes:
  - `set_graph_attr`, `get_graph_attr`
  - `set_node_attr`, `get_node_attr`
  - `set_edge_attr`, `get_edge_attr`

## `MultiDiGraph`

Implemented:
- Construction / basics:
  - `__len__`, `__contains__`, `__iter__`
  - `number_of_nodes`, `order`, `number_of_edges`, `size`
  - `is_directed`
- Core structure:
  - `add_node`, `add_nodes_from`
  - `add_edge` (supports `key`), `add_edges_from`
  - `remove_node`, `remove_edge(u,v,key)`
  - `has_node`, `has_edge`, `has_edge_key`
  - `nodes`, `successors`, `predecessors`, `neighbors`, `adj`
  - `succ_view`, `pred_view`
  - `out_degree`, `in_degree`, `degree`
  - `edges` (returns `List[(u,v,key)]`)
  - `clear`
  - `copy`, `subgraph`
- Traversal helpers:
  - `for_each_successor`, `for_each_predecessor`, `for_each_edge`
- Attributes:
  - `set_graph_attr`, `get_graph_attr`
  - `set_node_attr`, `get_node_attr`
  - `set_edge_attr`, `get_edge_attr`

---

# Parity gaps / deltas vs Python NetworkX

## Tier 0 (high priority)

Missing or different from NetworkX:
- `is_multigraph()` (NetworkX: `G.is_multigraph()`)
- `remove_nodes_from(nbunch)` (NetworkX silently ignores missing nodes)
- `remove_edges_from(ebunch)` (NetworkX silently ignores missing edges)
- Rich view/dict APIs:
  - `G.nodes` / `G.edges` views with `.data(...)`, iteration semantics
  - `G.adj` / `G.succ` / `G.pred` mapping-like interfaces
  - `G[u]` / `G[u][v]` / `G.get_edge_data(...)` semantics
- Full attribute dict compatibility (NetworkX returns mutable dict-like objects)
- Optional flags and overloads:
  - `copy(as_view=...)`
  - `subgraph(nbunch)` returning a view (NetworkX) vs a materialized copy here

## Tier 1 (medium priority)

Common algorithms and helpers not yet implemented (non-exhaustive):
- Undirected:
  - `is_tree`, `is_forest`, `has_path`
  - `connected_components`-adjacent: `node_connected_component`, `connected_component_subgraphs` (deprecated in NX, but common)
- Directed:
  - `has_path`
  - `is_weakly_connected`, `is_strongly_connected`
  - `topological_generations`
- General:
  - `shortest_path_length`, `single_source_shortest_path`, `single_source_dijkstra`

## Tier 2 (lower priority)

- Serialization / IO (`readwrite`), conversion (`to_undirected`, `to_directed`, `to_numpy_array`, etc.)
- Extensive algorithm suite (centrality, flows, isomorphism, communities, etc.)
- Subclassing / graph factory patterns (`Graph(...)` from data, `create_using`)

---

# Next planned work

Based on Tier 0 gaps above, the next concrete implementation steps are:
1. Add `is_multigraph()` to all graph types.
2. Add `remove_nodes_from(...)` and `remove_edges_from(...)` to all graph types, matching NetworkX semantics (ignore missing entries).
