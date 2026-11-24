"""Optional post-processing for the completed experiments (SciPy / Matplotlib)."""
import argparse
from bisect import bisect_right
import csv
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, median, quantiles

from experiment import META, NAMES, atomic_json, csv_file


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tests(out):
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon

    meta = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    if meta["mode"] != "run" or meta["profile"] not in ("official", "fast10"):
        raise ValueError("Report-level tests require a completed official or fast10 run.")
    all_rows = read_rows(out / "runs.csv")
    instances = sorted(meta["datasets"])
    seeds = set(meta["seeds"])
    blocks = []
    for name in instances:
        block = []
        for alg in META:
            rows = [r for r in all_rows if r["instance"] == name and r["algorithm"] == alg
                    and r["case"] == "default" and r["feasible"] == "True"]
            if len(rows) != len(seeds) or {int(r["seed"]) for r in rows} != seeds:
                raise ValueError(f"Incomplete or duplicate official rows: {name}, {alg}")
            # Use integer totals here to avoid subtracting nearly equal floating-point means.
            block.append(sum(int(r["best_value"]) for r in rows))
        blocks.append(block)
    ranks = [rankdata(block) for block in blocks]
    columns = list(zip(*blocks))
    all_ties = all(len(set(block)) == 1 for block in blocks)
    result = None if all_ties else friedmanchisquare(*columns)
    p = float(result.pvalue) if result is not None else None
    atomic_json(out / "friedman.json", {
        "statistic": float(result.statistic) if result is not None else None,
        "p_value": p, "alpha": 0.05, "n_instances": len(instances), "runs_per_algorithm_per_instance": len(seeds),
        "mean_ranks": {NAMES[a]: float(mean(r)) for a, r in zip(META, zip(*ranks))},
        "status": "all_blocks_tied" if all_ties else ("reject" if p < 0.05 else "not_rejected"),
        "unit_of_analysis": "one paired block per instance; algorithm score is mean color count over common seeds",
    })
    if p is not None and p < 0.05:
        pairs = []
        for i, j in combinations(range(len(META)), 2):
            d = [a - b for a, b in zip(columns[i], columns[j])]
            raw_p = float(wilcoxon(d).pvalue) if any(d) else 1.0
            pairs.append({"a": NAMES[META[i]], "b": NAMES[META[j]], "p_raw": raw_p,
                          "median_difference_in_colors_A_minus_B": median(d) / len(seeds)})
        pairs.sort(key=lambda row: row["p_raw"])
        prior = 0.0
        for i, row in enumerate(pairs):
            prior = min(1.0, max(prior, (len(pairs) - i) * row["p_raw"]))
            row["p_holm"] = prior
            row["reject_0_05"] = prior < 0.05
        csv_file(out / "wilcoxon_holm.csv", pairs)
    print(out / "friedman.json")


def plots(out, instance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    if meta["mode"] != "run":
        raise ValueError("Select a main or pilot run directory, not a sensitivity suite.")
    if instance not in meta["datasets"]:
        raise ValueError(f"Unknown dataset: {instance}")
    runs = [r for r in read_rows(out / "runs.csv") if r["instance"] == instance and r["case"] == "default"]
    trace = {}
    for r in read_rows(out / "convergence.csv"):
        if r["instance"] == instance and r["case"] == "default":
            trace.setdefault((r["algorithm"], r["seed"]), []).append((float(r["time_s"]), int(r["best"])))
    if len(trace) != len(runs) or not runs:
        raise ValueError(f"Missing convergence data for {instance}.")
    algorithms = [a for a in META if any(r["algorithm"] == a for r in runs)]
    horizon = float(meta["budgets"][instance])
    first = max(min(t for t, _ in log) for log in trace.values())
    if first >= horizon:
        raise ValueError("Budget is too short for all initial DSATUR solutions to be recorded before it ends.")
    times = [first + j * (horizon - first) / 199 for j in range(200)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for a in algorithms:
        aligned = []
        for (alg, _), events in trace.items():
            if alg != a:
                continue
            x, y = zip(*sorted(events, key=lambda e: e[0]))
            aligned.append([y[bisect_right(x, t) - 1] for t in times])
        samples = list(zip(*aligned))
        mids = [median(column) for column in samples]
        q = [quantiles(column, n=4, method="inclusive") if len(column) > 1
             else [column[0]]*3 for column in samples]
        line, = ax.step(times, mids, where="post", label=NAMES[a])
        ax.fill_between(times, [s[0] for s in q], [s[2] for s in q], alpha=0.12,
                        step="post", color=line.get_color())
    ax.set(title=f"Hội tụ: {instance}", xlabel="Thời gian (giây)", ylabel="Số màu hợp lệ tốt nhất")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / f"{instance}_convergence.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([[int(r["best_value"]) for r in runs if r["algorithm"] == a] for a in algorithms])
    ax.set_xticks(range(1, len(algorithms) + 1), [NAMES[a] for a in algorithms])
    ax.set(title=f"Phân bố kết quả: {instance}", ylabel="Số màu hợp lệ cuối cùng")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / f"{instance}_boxplot.png", dpi=160)
    plt.close(fig)
    print(f"Biểu đồ {instance}: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("stats", "plots"))
    p.add_argument("directory", type=Path)
    p.add_argument("--instance", default=None)
    args = p.parse_args()
    try:
        if args.action == "stats":
            tests(args.directory)
        else:
            if not args.instance:
                p.error("plots requires --instance")
            plots(args.directory, args.instance)
    except ImportError as e:
        p.exit(2, f"Cần cài {e.name}: python -m pip install scipy matplotlib\n")
    except (ValueError, OSError) as e:
        p.exit(2, f"Lỗi: {e}\n")


if __name__ == "__main__":
    main()
