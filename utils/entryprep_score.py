"""Summarize one complete, paired 16-route AV evaluation CSV."""
import argparse
import collections
import csv
from pathlib import Path


OUTCOMES = ("arrival", "collision", "timeout", "teleport", "insertion_failure", "wrong_exit")
ROUTES = tuple(f"route_{i:02}" for i in range(1, 17))


def score(rows, seeds):
    expected = {(route, seed) for route in ROUTES for seed in seeds}
    seen = set()
    counts = {r: collections.Counter() for r in ROUTES}
    for row in rows:
        route = row["route"]
        pair = route, int(row["seed"])
        if pair not in expected or pair in seen or row["outcome"] not in OUTCOMES:
            raise ValueError(f"Unexpected, duplicate or invalid evaluation case: {row}")
        if int(row.get("route_overrides", 0)):
            raise ValueError(f"Policy action overridden: {pair}")
        seen.add(pair)
        counts[route][row["outcome"]] += 1
    if seen != expected:
        raise ValueError(f"{len(expected - seen)} required cases are missing")

    report = []
    for route, count in counts.items():
        n = sum(count.values())
        meets_rates = count["arrival"] * 100 >= 95 * n and count["collision"] * 100 < 5 * n
        report.append(dict(route=route, runs=n, **{k: count[k] for k in OUTCOMES},
                           arrival_pct=round(100 * count["arrival"] / n, 3),
                           collision_pct=round(100 * count["collision"] / n, 3),
                           rates_pass=meets_rates))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=30001)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("Provide at least one seed")
    with args.evaluation.open(encoding="utf8", newline="") as source:
        report = score(csv.DictReader(source), range(args.first_seed, args.first_seed + args.seeds))
    if args.output.resolve() == args.evaluation.resolve():
        parser.error("Choose a separate path for the route report")
    with args.output.open("w", encoding="utf8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    print(f"{sum(r['rates_pass'] for r in report)}/{len(report)} routes meet the two rate thresholds")


if __name__ == "__main__":
    main()
