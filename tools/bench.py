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


def _ensure_mojo_bench_built(repo_root: Path, *, force: bool) -> Path:
    mojo = repo_root / ".venv" / "bin" / "mojo"
    bench = repo_root / "benches" / "graph_bench.mojo"

    out_dir = repo_root / ".bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / "graph_bench"

    needs_build = force or (not exe.exists())
    if not needs_build:
        needs_build = exe.stat().st_mtime < _latest_mojo_mtime(repo_root)

    if needs_build:
        cmd = [str(mojo), "build", "-I", str(repo_root), "-o", str(exe), str(bench)]
        subprocess.run(cmd, cwd=repo_root, check=True)

    return exe


def _run_mojo_bench(repo_root: Path, n: int, m: int, reps: int, *, force_build: bool, skip_build: bool) -> dict:
    if skip_build:
        exe = repo_root / ".bench" / "graph_bench"
    else:
        exe = _ensure_mojo_bench_built(repo_root, force=force_build)

    cmd = [str(exe), str(n), str(m), str(reps)]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)

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
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-file", type=str, default="bench.png")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    mojo = _run_mojo_bench(
        repo_root,
        args.n,
        args.m,
        args.reps,
        force_build=args.force_build,
        skip_build=args.skip_build,
    )
    py = _run_python_networkx_bench(args.n, args.m, args.reps)

    result = {
        "n": args.n,
        "m": args.m,
        "reps": args.reps,
        **mojo,
        **py,
    }
    if args.plot:
        import matplotlib.pyplot as plt  # type: ignore

        mojo_avg = int(result["mojo_build_ns_avg"])
        py_avg = int(result["python_build_ns_avg"])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["mojo", "python networkx"], [mojo_avg, py_avg])
        ax.set_ylabel("avg build time (ns)")
        ax.set_title(f"n={args.n} m={args.m} reps={args.reps}")
        fig.tight_layout()
        fig.savefig(repo_root / args.plot_file)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
