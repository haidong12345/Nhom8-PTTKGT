"""Audit fast10 output and report compact, reproducible aggregate figures."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys

from experiment import HERE, META, NAMES, OFFICIAL, read_graph, validate_files


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify(out):
    meta = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    assert meta["profile"] == "fast10" and meta["seeds"] == list(range(10))
    rows = read_csv(out / "runs.csv")
    keys = Counter((r["instance"], r["algorithm"], int(r["seed"])) for r in rows)
    assert keys == Counter((inst, algo, seed) for inst in OFFICIAL for algo in META for seed in range(10))
    assert len(list((out / "checkpoints").glob("*.json"))) == len(rows) == 720
    assert len(list((out / "solutions").glob("*.sol"))) == len(rows)
    references = {r["instance"]: r for r in read_csv(out / "instances.csv")}
    summary = read_csv(out / "table_12_7.csv")
    assert Counter((r["instance"], r["algorithm"]) for r in summary) == Counter(
        (inst, NAMES[a]) for inst in OFFICIAL for a in META)
    cols = {(r["instance"], r["algorithm"]): r for r in summary}
    performance = Counter()
    minimum_avg = Counter()
    for inst in OFFICIAL:
        graph = HERE / "data" / f"{inst}.col"
        adj, _ = read_graph(graph)
        ref = int(references[inst]["reference"])
        assert meta["budgets"][inst] == (2 if len(adj) >= 250 else 0.5)
        avgs = {}
        for a in META:
            subset = [r for r in rows if r["instance"] == inst and r["algorithm"] == a]
            row = cols[(inst, NAMES[a])]
            values = []
            for r in subset:
                assert r["case"] == "default" and r["feasible"] == "True"
                assert float(r["budget_s"]) == meta["budgets"][inst]
                sol = out / "solutions" / f"{inst}__{a}__default__{r['seed']}.sol"
                v = validate_files(graph, sol)
                assert v["feasible"] and v["used_colors"] == int(r["best_value"])
                values.append(v["used_colors"])
            avg = statistics.mean(values)
            hits = sum(v == ref for v in values)
            assert int(row["runs"]) == len(values) == 10
            assert int(row["best"]) == min(values) and abs(float(row["avg"]) - avg) < 1e-6
            assert int(row["reference_hits"]) == hits
            performance[a] += hits
            avgs[a] = avg
        minimum_avg.update(a for a, avg in avgs.items() if avg == min(avgs.values()))
        print(f"{inst} ({references[inst]['reference_type']}, ref={ref}): " +
              ", ".join(f"{NAMES[a]} {avgs[a]:.1f}" for a in META))
    print(f"Validated {len(rows)} solution files and {len(summary)} aggregate rows.")
    print("Reference hits (each out of 120):", {NAMES[a]: performance[a] for a in META})
    print("Instances with minimum average, counting ties:", {NAMES[a]: minimum_avg[a] for a in META})
    print("Total solver seconds:", round(sum(float(r["time_s"]) for r in rows), 3))


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
