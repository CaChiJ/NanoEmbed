#!/usr/bin/env python3
"""Run all bench scenarios for a given milestone, write an aggregated JSON.

Each scenario shells out to nanoembed-bench (a single self-contained tool
that prints one scenario's JSON). This script collects them into a single
file under bench/results/.
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict, List


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def git_sha() -> str:
    try:
        return (subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=repo_root())
                .decode().strip())
    except Exception:
        return "unknown"


def build_cmd(bench: pathlib.Path, sc: Dict[str, Any], root: pathlib.Path) -> List[str]:
    cmd: List[str] = [
        str(bench),
        "--model",     str(root / sc["model"]),
        "--inputs",    str(root / sc["inputs"]),
        "--scenario",  sc["name"],
        "--warmup",    str(sc.get("warmup", 5)),
        "--iter",      str(sc.get("iter", 50)),
        "--threads",   str(sc.get("threads", 0)),
    ]
    if sc.get("pooling") == "cls":
        cmd.append("--cls")
    if not sc.get("normalize", True):
        cmd.append("--no-normalize")
    if sc.get("max_seq_len", 0) > 0:
        cmd += ["--max-seq-len", str(sc["max_seq_len"])]
    # Sampler cadences: RSS is cheap to poll, smaps_rollup walks the page table
    # under the worker's mmap_lock and must stay slow. Overridable per scenario
    # for cases where a short run needs more PSS/USS samples to be meaningful.
    if sc.get("rss_interval_ms", 0) > 0:
        cmd += ["--rss-interval-ms", str(sc["rss_interval_ms"])]
    if sc.get("rollup_interval_ms", 0) > 0:
        cmd += ["--rollup-interval-ms", str(sc["rollup_interval_ms"])]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", required=True, help="e.g. M3, M4 ...")
    parser.add_argument("--scenarios", type=pathlib.Path,
                        default=pathlib.Path("bench/scenarios.yaml"))
    parser.add_argument("--bench", type=pathlib.Path,
                        default=pathlib.Path("build/bin/nanoembed-bench"))
    parser.add_argument("--out", type=pathlib.Path,
                        help="defaults to bench/results/<git_sha>.json")
    parser.add_argument("--filter",
                        help="run only the scenario with this name")
    args = parser.parse_args()

    try:
        import yaml  # type: ignore
    except ImportError:
        print("error: pip install -r requirements-dev.txt (need pyyaml)",
              file=sys.stderr)
        return 2

    root = repo_root()
    cfg_path = (args.scenarios if args.scenarios.is_absolute()
                else root / args.scenarios)
    cfg = yaml.safe_load(cfg_path.read_text())
    scenarios: List[Dict[str, Any]] = cfg.get("scenarios", [])

    bench_bin = (args.bench if args.bench.is_absolute()
                 else root / args.bench)
    if not bench_bin.exists():
        print(f"error: bench binary not found at {bench_bin}; "
              "build the project first (cmake --build build)",
              file=sys.stderr)
        return 2

    sha = git_sha()
    out_path = args.out or (root / "bench/results" / f"{sha}.json")

    aggregated: Dict[str, Any] = {
        "milestone": args.milestone,
        "git_sha":   sha,
        "scenarios": {},
    }

    for sc in scenarios:
        if args.milestone not in sc.get("milestones", []):
            continue
        if args.filter and sc["name"] != args.filter:
            continue

        print(f"==> {sc['name']}", file=sys.stderr)
        cmd = build_cmd(bench_bin, sc, root)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"FAIL: {sc['name']}\n{proc.stderr}", file=sys.stderr)
            return 1
        aggregated["scenarios"][sc["name"]] = json.loads(proc.stdout)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2) + "\n")
    print(f"wrote {out_path} with {len(aggregated['scenarios'])} scenarios",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
