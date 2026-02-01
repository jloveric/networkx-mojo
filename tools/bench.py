#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _latest_mojo_mtime(repo_root: Path) -> float:
    latest = 0.0
    for p in (repo_root / "networkx").rglob("*.mojo"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except FileNotFoundError:
            continue
    bench = repo_root / "benches" / "graph_bench.mojo"
    latest = max(latest, bench.stat().st_mtime)
    return latest


def _ensure_mojo_bench_built(repo_root: Path, *, force: bool, auto_rebuild: bool) -> Path:
    mojo = repo_root / ".venv" / "bin" / "mojo"
    bench = repo_root / "benches" / "graph_bench.mojo"

    out_dir = repo_root / ".bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / "graph_bench"

    needs_build = force or (not exe.exists())
    if not needs_build and auto_rebuild:
        needs_build = exe.stat().st_mtime < _latest_mojo_mtime(repo_root)

    if needs_build:
        build_start = time.perf_counter_ns()
        print("[bench] building mojo benchmark...", file=sys.stderr)
        cmd = [str(mojo), "build", "-I", str(repo_root), "-o", str(exe), str(bench)]
        subprocess.run(cmd, cwd=repo_root, check=True)
        build_end = time.perf_counter_ns()
        print(f"[bench] build done in {(build_end - build_start) / 1e9:.3f}s", file=sys.stderr)

    return exe


def _run_mojo_bench(
    repo_root: Path,
    n: int,
    m: int,
    reps: int,
    *,
    force_build: bool,
    skip_build: bool,
    auto_rebuild: bool,
) -> dict:
    if skip_build:
        exe = repo_root / ".bench" / "graph_bench"
    else:
        exe = _ensure_mojo_bench_built(repo_root, force=force_build, auto_rebuild=auto_rebuild)

    if not exe.exists():
        raise RuntimeError(
            "Mojo benchmark executable not found. Run without --skip-build to build it, "
            "or run with --force-build."
        )

    cmd = [str(exe), str(n), str(m), str(reps)]
    run_start = time.perf_counter_ns()
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
    run_end = time.perf_counter_ns()
    print(f"[bench] mojo run done in {(run_end - run_start) / 1e9:.6f}s", file=sys.stderr)

    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _run_python_networkx_bench(n: int, m: int, reps: int) -> dict:
    # Avoid importing the local Mojo package directory named `networkx`.
    # We want the Python `networkx` package.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path = [
        p
        for p in sys.path
        if p
        and p not in {".", ""}
        and Path(p).resolve() != repo_root
    ]

    import networkx as py_nx  # type: ignore

    print("[bench] running python networkx benchmark...", file=sys.stderr)
    py_start = time.perf_counter_ns()
    total_ns = 0
    for _ in range(reps):
        g = py_nx.Graph()
        start = time.perf_counter_ns()
        for i in range(m):
            u = i % n
            v = (i * 9973 + 17) % n
            if v == u:
                v = (v + 1) % n
            g.add_edge(u, v)
        end = time.perf_counter_ns()
        total_ns += end - start

    py_end = time.perf_counter_ns()
    print(f"[bench] python run done in {(py_end - py_start) / 1e9:.6f}s", file=sys.stderr)

    return {
        "python_build_ns_total": total_ns,
        "python_build_ns_avg": total_ns // reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--m", type=int, default=500)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--auto-rebuild", action="store_true")
    parser.add_argument("--compare-python", action="store_true")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-file", type=str, default="bench.png")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    result: dict[str, object] = {
        "n": args.n,
        "m": args.m,
        "reps": args.reps,
    }

    if not args.python_only:
        mojo = _run_mojo_bench(
            repo_root,
            args.n,
            args.m,
            args.reps,
            force_build=args.force_build,
            skip_build=args.skip_build,
            auto_rebuild=args.auto_rebuild,
        )
        result.update(mojo)

    if args.compare_python or args.python_only:
        py = _run_python_networkx_bench(args.n, args.m, args.reps)
        result.update(py)

    if args.plot:
        import matplotlib.pyplot as plt  # type: ignore

        labels = []
        values = []

        if "mojo_build_ns_avg" in result:
            labels.append("mojo")
            values.append(int(result["mojo_build_ns_avg"]))
        if "python_build_ns_avg" in result:
            labels.append("python networkx")
            values.append(int(result["python_build_ns_avg"]))

        if len(labels) == 0:
            raise RuntimeError("No benchmark results available to plot")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values)
        ax.set_ylabel("avg build time (ns)")
        ax.set_title(f"n={args.n} m={args.m} reps={args.reps}")
        fig.tight_layout()
        fig.savefig(repo_root / args.plot_file)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
