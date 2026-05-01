#!/usr/bin/env python3
"""Dump a tiny eval corpus from MTEB / STS-B for use as a regression fixture.

Output: tests/corpus/eval.txt — one sentence per line.

This is run once at project setup; the resulting corpus is committed.
Re-run only when changing the corpus composition.
"""

import argparse
import pathlib
import random
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="number of sentences to sample")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducibility")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("tests/corpus/eval.txt"))
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("error: install dev deps first ("
              "pip install -r requirements-dev.txt)", file=sys.stderr)
        return 2

    # MTEB / STS-B test split — Apache 2.0, well-known.
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")

    # Pull both columns to widen the pool, then shuffle/sample.
    pool = list(set(ds["sentence1"]) | set(ds["sentence2"]))
    pool = [s.strip() for s in pool if s and s.strip()]
    pool.sort()  # deterministic before shuffle

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sampled = pool[: args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s in sampled:
            f.write(s + "\n")

    print(f"wrote {len(sampled)} sentences to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
