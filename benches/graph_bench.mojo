from networkx import Graph
from sys import argv
from time import perf_counter_ns


fn main() raises:
    var args = argv()

    var n = 200
    var m = 500
    var reps = 1

    if len(args) > 1:
        n = Int(args[1])
    if len(args) > 2:
        m = Int(args[2])
    if len(args) > 3:
        reps = Int(args[3])

    var total: UInt = 0

    for _ in range(reps):
        var g = Graph[Int]()
        var start = perf_counter_ns()
        for i in range(m):
            var u = i % n
            var v = (i * 9973 + 17) % n
            if v == u:
                v = (v + 1) % n
            g.add_edge(u, v)
        var end = perf_counter_ns()
        total += end - start

    print("mojo_build_ns_total=", total)
    print("mojo_build_ns_avg=", total // UInt(reps))
