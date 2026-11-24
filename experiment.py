#!/usr/bin/env python3
"""Run reproducible graph-coloring experiments on one machine, sequentially."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
from statistics import mean, stdev
from time import perf_counter
from urllib.request import urlopen

from de06 import clique_lower_bound, dsatur, exact_bb, gen_gnp, normalize, welsh_powell
from solvers import DEFAULTS, KERNELS, Meter, hill_climb, reduce_one, solve_fixed

HERE = Path(__file__).resolve().parent
DATA_URL = "https://mat.tepper.cmu.edu/COLOR/instances/"
SOURCE_PAGE = "https://mat.tepper.cmu.edu/COLOR/instances.html"
# name: (vertices, edges, chromatic number)
STANDARD = {"myciel3": (11, 20, 4), "myciel4": (23, 71, 5), "myciel5": (47, 236, 6),
            "queen5_5": (25, 160, 5), "queen6_6": (36, 290, 7), "queen8_8": (64, 728, 9)}
SYNTHETIC = {f"gnp_{n}_{tag}": (n, p, 6024 + n + tag)
             for n in (30, 100, 250) for tag, p in ((10, 0.1), (50, 0.5))}
TUNING = {f"tune_{n}_{tag}": (n, p, 8491 + n + tag)
          for n in (30, 100, 250) for tag, p in ((15, 0.15), (45, 0.45))}
SMALL = ("vidu", "myciel3", "myciel4")
OFFICIAL = tuple(STANDARD) + tuple(SYNTHETIC)
PILOT = ("myciel3", "queen5_5", "gnp_30_50")
META = tuple(KERNELS)
NAMES = dict(zip(META, ("TabuCol", "SA", "GA", "ACO", "VNS", "HEA")))
EXAMPLE = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 6),
           (4, 5), (4, 7), (5, 6), (5, 7), (6, 7)]
SENSITIVITY = {
    "tabu": {"lam": (0.3, 0.6, 0.9)}, "sa": {"alpha": (0.90, 0.95, 0.99)},
    "ga": {"population": (10, 20, 40), "mutation_scale": (1, 2, 5)},
    "aco": {"rho": (0.1, 0.3, 0.5)}, "vns": {"s_max": (2, 4, 8), "vnd_scale": (0.5, 1, 2)},
    "hea": {"local_scale": (1, 3, 10)},
}


def read_graph(path):
    """Read ASCII DIMACS; ignore duplicate edges, reject loops and malformed records."""
    header, edges, raw = None, set(), 0
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts or parts[0] == "c":
            continue
        try:
            if parts[0] == "p" and len(parts) == 4 and parts[1] in ("edge", "col") and header is None:
                n, m = map(int, parts[2:])
                if n < 1 or m < 0:
                    raise ValueError("n or m out of range")
                header = n, m
            elif parts[0] == "e" and len(parts) == 3 and header is not None:
                u, v = map(int, parts[1:])
                if not (1 <= u <= header[0] and 1 <= v <= header[0]) or u == v:
                    raise ValueError("loop or vertex out of range")
                edges.add(tuple(sorted((u - 1, v - 1))))
                raw += 1
            else:
                raise ValueError("invalid DIMACS record")
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if header is None or header[1] not in (raw, len(edges)):
        raise ValueError(f"{path}: missing header or incorrect edge count")
    adj = [set() for _ in range(header[0])]
    for u, v in sorted(edges):
        adj[u].add(v)
        adj[v].add(u)
    return adj, sorted(edges)


def verdict(edges, n, k, col):
    """Independent of State and all solvers; return justification and every bad edge."""
    errors = []
    if len(col) != n:
        errors.append(f"Expected {n} labels, got {len(col)}.")
    if not 1 <= k <= n or any(c < 0 or c >= k for c in col):
        errors.append(f"k or labels outside their declared range (1..{k} in .sol).")
    used = len(set(col))
    if used != k:
        errors.append(f"Declared {k} colors, actual {used}.")
    bad = [(u + 1, v + 1) for u, v in edges if v < len(col) and col[u] == col[v]]
    return {"feasible": not errors and not bad, "used_colors": used, "conflicts": bad, "errors": errors}


def validate_files(graph, solution):
    adj, edges = read_graph(graph)
    rows = Path(solution).read_text(encoding="utf-8").strip().splitlines()
    if len(rows) != 2 or len(rows[0].split()) != 1:
        return {"feasible": False, "errors": [".sol must have a k line and a color-label line."]}
    try:
        k = int(rows[0])
        col = [int(c) - 1 for c in rows[1].split()]
    except ValueError:
        return {"feasible": False, "errors": ["Color labels and k must be integers."]}
    return verdict(edges, len(adj), k, col)


def save_graph(path, n, edges, comment):
    path.write_text(f"c {comment}\np edge {n} {len(edges)}\n" +
                    "".join(f"e {u+1} {v+1}\n" for u, v in edges), encoding="utf-8")


def data_for(names, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in names:
        path = data_dir / f"{name}.col"
        if name in STANDARD and not path.exists():
            try:
                with urlopen(f"{DATA_URL}{name}.col", timeout=20) as response:
                    payload = response.read()
                tmp = path.with_suffix(".download")
                tmp.write_bytes(payload)
                adj, edges = read_graph(tmp)
                if (len(adj), len(edges)) != STANDARD[name][:2]:
                    raise ValueError("Downloaded graph shape mismatches CMU reference.")
                tmp.replace(path)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot prepare {name}. Download {DATA_URL}{name}.col to {path}: {exc}") from exc
        elif name == "vidu" and not path.exists():
            save_graph(path, 7, [(u - 1, v - 1) for u, v in EXAMPLE], "Seven-vertex example from the assignment.")
        elif not path.exists():
            n, p, seed = (SYNTHETIC | TUNING)[name]
            _, edges = gen_gnp(n, p, seed)
            save_graph(path, n, edges, f"gnp n={n} p={p} data_seed={seed}")
        adj, edges = read_graph(path)
        if name in STANDARD and (len(adj), len(edges)) != STANDARD[name][:2]:
            raise ValueError(f"{path}: graph shape differs from published DIMACS data")
        records[name] = (path, adj, edges)
    return records


def budget_for(n):
    return 30 if n <= 125 else (60 if n < 250 else 120)


def fast_budget_for(n):
    return 0.5 if n <= 125 else (1.0 if n < 250 else 2.0)


def checked_coloring(adj, edges, color):
    checked = verdict(edges, len(adj), len(set(color)), color)
    if not checked["feasible"]:
        raise RuntimeError(f"Solver gave an invalid coloring: {checked}")
    return checked["used_colors"]


def launch_method(name, adj, edges, seconds, seed, overrides):
    start = perf_counter()
    meter = Meter(start + seconds)
    rng = random.Random(seed)
    best = normalize(dsatur(adj))
    best_k = len(set(best))
    t_best = perf_counter() - start
    events = [{"time_s": t_best, "iter": 0, "evals": 0, "best": best_k,
               "full_evals": 0, "delta_evals": 0}]
    lower = len(clique_lower_bound(adj)) if meter.open() else 1
    attempts = []
    while best_k > lower and meter.open():
        k = best_k - 1
        candidate, conflict = solve_fixed(name, adj, k, reduce_one(adj, best, k, rng), rng, meter, overrides)
        actual_conflicts = sum(candidate[u] == candidate[v] for u, v in edges)
        if len(candidate) != len(adj) or not all(0 <= a < k for a in candidate) or conflict != actual_conflicts:
            raise RuntimeError(f"{name} returned an inconsistent fixed-k state")
        attempts.append({"target": k, "conflicts": conflict})
        if conflict:
            break
        best = normalize(candidate)
        best_k = len(set(best))
        t_best = perf_counter() - start
        events.append({"time_s": t_best, "iter": meter.steps, "evals": meter.full + meter.delta,
                       "best": best_k, "full_evals": meter.full, "delta_evals": meter.delta})
    elapsed = perf_counter() - start
    verified_k = checked_coloring(adj, edges, best)
    return {
        "best_value": verified_k, "time_s": elapsed, "time_to_best_s": t_best, "feasible": True,
        "evaluations": meter.full + meter.delta, "full_evaluations": meter.full, "delta_evaluations": meter.delta,
        "iterations": meter.steps, "lower_bound": lower, "fixed_k_attempts": len(attempts),
        "attempts": attempts, "solution": best, "events": events, "budget_s": seconds,
    }


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def csv_file(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as out:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def source_fingerprint():
    return {name: sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("de06.py", "solvers.py", "experiment.py")}


def hardware():
    cpu = platform.processor() or platform.machine()
    ram = None
    cpuinfo = Path("/proc/cpuinfo")
    meminfo = Path("/proc/meminfo")
    if cpuinfo.is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    if meminfo.is_file():
        ram = int(meminfo.read_text().splitlines()[0].split()[1]) * 1024
    return {"platform": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine(), "cpu": cpu, "host_memory_bytes_reported": ram,
            "hostname": platform.node()}


def export_runs(out, saved, references):
    raw, logs, attempts, group = [], [], [], {}
    for row in saved:
        raw.append({k: v for k, v in row.items() if k not in ("solution", "events", "attempts")})
        key = row["instance"], row["algorithm"], row["case"]
        group.setdefault(key, []).append(row)
        identity = {k: row[k] for k in ("instance", "algorithm", "case", "seed", "budget_s")}
        logs.extend(identity | event for event in row["events"])
        attempts.extend(identity | {"trial": i} | event for i, event in enumerate(row["attempts"], 1))
    csv_file(out / "runs.csv", raw)
    csv_file(out / "convergence.csv", logs)
    csv_file(out / "fixed_k_attempts.csv", attempts)
    csv_file(out / "instances.csv", [{"instance": name} | ref for name, ref in references.items()])
    rows = []
    for (inst, name, case), runs in group.items():
        values = [r["best_value"] for r in runs]
        optimum = references[inst]["reference"]
        rows.append({"instance": inst, "reference": optimum,
                     "reference_type": references[inst]["reference_type"],
                     "algorithm": NAMES[name], "case": case, "runs": len(values),
                     "best": min(values), "avg": round(mean(values), 6),
                     "std": round(stdev(values), 6) if len(values) > 1 else "",
                     "gap_best_pct": round(100 * abs(min(values) - optimum) / optimum, 6),
                     "gap_avg_pct": round(100 * abs(mean(values) - optimum) / optimum, 6),
                     "reference_hits": sum(v == optimum for v in values),
                     "avg_time_s": round(mean(r["time_s"] for r in runs), 6),
                     "avg_time_to_best_s": round(mean(r["time_to_best_s"] for r in runs), 6)})
    csv_file(out / "table_12_7.csv", rows)


def configurations(args):
    if args.mode == "run":
        return [(alg, "default", {}) for alg in args.algorithms]
    return [(alg, f"{field}={value}", {field: value})
            for alg in args.algorithms for field, levels in SENSITIVITY[alg].items() for value in levels]


def run_suite(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.profile == "fast10" and args.mode != "run":
        raise ValueError("--profile fast10 applies only to the main run.")
    strict = args.mode == "run" and args.profile in ("official", "fast10")
    preset_runs = {"pilot": 2, "official": 30, "fast10": 10}[args.profile]
    names = tuple(dict.fromkeys(args.instances or (
        TUNING if args.mode == "sensitivity" else (OFFICIAL if strict else PILOT))))
    if args.mode == "sensitivity" and any(name not in TUNING for name in names):
        raise ValueError("Sensitivity runs must use tune_* instances separate from the main experiment.")
    data = data_for(names, args.data)
    runs = args.runs if args.runs is not None else preset_runs
    if runs < 1:
        raise ValueError("At least one seed is required.")
    settings = configurations(args)
    if strict and (runs != preset_runs or args.seconds is not None or args.seed_start != 0
                   or names != OFFICIAL or tuple(args.algorithms) != META):
        raise ValueError(f"The {args.profile} preset fixes 12 datasets, 6 algorithms, "
                         f"seeds 0..{preset_runs-1}, and per-size budgets. Use pilot to customize.")
    budgets = {"pilot": lambda n: 0.25, "official": budget_for, "fast10": fast_budget_for}
    per_budget = {name: (args.seconds if args.seconds is not None else budgets[args.profile](len(data[name][1])))
                  for name in names}
    meta = {"mode": args.mode, "profile": args.profile, "machine": hardware(), "code": source_fingerprint(),
            "seeds": list(range(args.seed_start, args.seed_start + runs)),
            "budgets": per_budget, "defaults": DEFAULTS, "settings": [list(v) for v in settings],
            "datasets": {name: sha256(data[name][0].read_bytes()).hexdigest() for name in names}}
    metadata = out / "environment.json"
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != meta:
            raise ValueError("Output folder already has a different experiment; use another --out.")
    else:
        atomic_json(metadata, meta)
    exact_refs = {}
    if args.exact_csv:
        with args.exact_csv.open(encoding="utf-8-sig", newline="") as src:
            for row in csv.DictReader(src):
                name = row["instance"]
                if (row["proven"] == "True" and name in data
                        and row["graph_sha256"] == meta["datasets"][name]):
                    exact_refs[name] = int(row["colors"])
    checkpoints, sols = out / "checkpoints", out / "solutions"
    checkpoints.mkdir(exist_ok=True)
    sols.mkdir(exist_ok=True)
    total = len(names) * runs * len(settings)
    print(f"{args.profile}: {total} lượt, chạy tuần tự. CSV dựng lại từ checkpoints khi kết thúc.", flush=True)
    saved = []
    for inst in names:
        _, adj, edges = data[inst]
        for seed_index, seed in enumerate(meta["seeds"], 1):
            # Randomize the run order reproducibly without changing any solver seed.
            order = settings[:]
            random.Random(f"{inst}:{seed}").shuffle(order)
            for alg, case, overrides in order:
                key = f"{inst}__{alg}__{case}__{seed}"
                checkpoint = checkpoints / f"{key}.json"
                if checkpoint.exists():
                    row = json.loads(checkpoint.read_text(encoding="utf-8"))
                    checked_coloring(adj, edges, row["solution"])
                else:
                    row = launch_method(alg, adj, edges, per_budget[inst], seed, overrides)
                    row.update(instance=inst, n=len(adj), m=len(edges), algorithm=alg, case=case,
                               seed=seed, run=seed - args.seed_start + 1,
                               params=json.dumps(DEFAULTS[alg] | overrides, sort_keys=True))
                    atomic_json(checkpoint, row)
                col = row["solution"]
                (sols / f"{key}.sol").write_text(
                    f'{row["best_value"]}\n' + " ".join(str(c + 1) for c in col) + "\n", encoding="utf-8")
                saved.append(row)
            if args.profile == "fast10" and seed_index % 2 == 0:
                print(f"  {inst} seed {seed_index}/{runs}: {len(saved)}/{total}", flush=True)
        print(f"  {inst}: {len(saved)}/{total}", flush=True)
    references = {}
    for inst, (path, adj, edges) in data.items():
        internal = min(r["best_value"] for r in saved if r["instance"] == inst)
        if inst in STANDARD or inst == "vidu":
            reference, kind = (3, "assignment_optimum") if inst == "vidu" else (STANDARD[inst][2], "published_optimum")
        elif inst in exact_refs:
            reference, kind = exact_refs[inst], "proven_optimum"
        else:
            reference, kind = internal, "best_observed_not_proven"
        if internal < reference:
            raise RuntimeError(f"{inst}: feasible solution beats known optimum; check input/reference")
        references[inst] = {"vertices": len(adj), "edges": len(edges), "reference": reference,
                            "reference_type": kind, "file": path.name,
                            "source": SOURCE_PAGE if inst in STANDARD else (
                                "Đề bài" if inst == "vidu" else f"G(n,p), data seed={(SYNTHETIC | TUNING)[inst][2]}"),
                            "graph_sha256": meta["datasets"][inst]}
    export_runs(out, saved, references)
    print(f"Xuất kết quả: {out / 'table_12_7.csv'}")


def baseline(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = data_for(args.instances or SMALL, args.data)
    exact_rows, heur_rows, local_rows = [], [], []
    for name, (path, adj, edges) in data.items():
        k_ref = 3 if name == "vidu" else STANDARD.get(name, (None, None, None))[2]
        start = perf_counter()
        col, k, optimal, lb, nodes = exact_bb(adj, args.exact_seconds)
        elapsed = perf_counter() - start
        if checked_coloring(adj, edges, normalize(col)) != k or (optimal and k_ref is not None and k != k_ref):
            raise RuntimeError(f"{name}: inconsistent exact search result")
        exact_rows.append(dict(instance=name, colors=k, proven=optimal, lower_bound=lb,
                               nodes=nodes, time_s=elapsed, target_limit_s=args.exact_seconds,
                               graph_sha256=sha256(path.read_bytes()).hexdigest()))
        for method in (welsh_powell, dsatur):
            start = perf_counter()
            col = normalize(method(adj))
            elapsed = perf_counter() - start
            heur_rows.append(dict(instance=name, algorithm=method.__name__,
                                  colors=checked_coloring(adj, edges, col), time_s=elapsed))
        # Both hill climbers start from the same k and identical assignments for each seed.
        start_col = normalize(dsatur(adj))
        target = max(1, len(set(start_col)) - 1)
        for seed in range(args.seed_start, args.seed_start + (args.runs or 1)):
            if len(set(start_col)) > 1:
                initial = reduce_one(adj, start_col, target, random.Random(seed))
            else:
                initial = start_col[:]
            initial_conflicts = sum(initial[u] == initial[v] for u, v in edges)
            for first in (True, False):
                start = perf_counter()
                meter = Meter(start + (args.seconds if args.seconds is not None else budget_for(len(adj))))
                col, f = hill_climb(adj, target, initial, random.Random(seed), meter, first)
                elapsed = perf_counter() - start
                if f != sum(col[u] == col[v] for u, v in edges) or f > initial_conflicts:
                    raise RuntimeError("Inconsistent hill-climbing delta.")
                local_rows.append(dict(instance=name, algorithm="HC-FI" if first else "HC-BI",
                                       seed=seed, target_k=target, conflicts_initial=initial_conflicts,
                                       conflicts_final=f, accepted_moves=meter.steps,
                                       delta_evaluations=meter.delta, time_s=elapsed))
        print(f"{name}: {k} màu; chứng minh tối ưu: {optimal}; {nodes} nút", flush=True)
        csv_file(out / "exact.csv", exact_rows)
        csv_file(out / "constructive.csv", heur_rows)
        csv_file(out / "local_search.csv", local_rows)
    atomic_json(out / "baseline_environment.json", {"machine": hardware(), "code": source_fingerprint()})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "run", "sensitivity", "baseline", "validate"):
        cmd = sub.add_parser(mode)
        if mode == "validate":
            cmd.add_argument("graph", type=Path)
            cmd.add_argument("solution", type=Path)
            continue
        cmd.add_argument("--data", type=Path, default=HERE / "data")
        cmd.add_argument("--instances", nargs="+", choices=(*SMALL[:1], *OFFICIAL, *TUNING))
        if mode == "prepare":
            continue
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--seconds", type=float)
        cmd.add_argument("--runs", type=int)
        cmd.add_argument("--seed-start", type=int, default=0)
        if mode == "baseline":
            cmd.add_argument("--exact-seconds", type=float, default=600)
        else:
            cmd.add_argument("--profile", choices=("pilot", "official", "fast10"), default="pilot")
            cmd.add_argument("--algorithms", nargs="+", default=META, choices=META)
            cmd.add_argument("--exact-csv", type=Path)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        if any(value is not None and value <= 0 for value in (
                getattr(args, "seconds", None), getattr(args, "exact_seconds", None), getattr(args, "runs", None))):
            p.error("time budgets and run counts must be positive")
        if args.mode == "prepare":
            prepared = data_for(args.instances or ("vidu", *OFFICIAL, *TUNING), args.data)
            for name, (path, adj, edges) in prepared.items():
                print(f"{name}: {len(adj)} đỉnh, {len(edges)} cạnh")
        elif args.mode == "validate":
            status = validate_files(args.graph, args.solution)
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0 if status["feasible"] else 1
        elif args.mode == "baseline":
            baseline(args)
        else:
            run_suite(args)
    except (OSError, ValueError) as exc:
        p.exit(2, f"Lỗi: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
