#!/usr/bin/env python3
"""Dump sentence-transformers reference embeddings as a binary fixture.

The C++ integration test loads this fixture and asserts cosine similarity
≥ 0.9999 (per-sample) and ≥ 0.99999 (mean) against our embedder output.

Output format ("NEGD" multi-sample binary):
    "NEGD" (4 bytes)
    version (u32, =1)
    n_samples (u32)
    n_embed   (u32)
    For each sample:
        text_len (u32) + text (utf-8)
        embedding (f32[n_embed])
"""

import argparse
import pathlib
import struct
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--corpus", type=pathlib.Path,
                        default=pathlib.Path("tests/corpus/eval.txt"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("tests/fixtures/golden/st-bge-small.bin"))
    parser.add_argument("--no-normalize", action="store_true",
                        help="disable L2 normalization (off by default; "
                             "sentence-transformers normalizes by default for BGE)")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("error: pip install -r requirements-dev.txt (need sentence-transformers)",
              file=sys.stderr)
        return 2

    model = SentenceTransformer(args.model)

    with args.corpus.open(encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]

    embeddings = model.encode(
        texts,
        normalize_embeddings=not args.no_normalize,
        convert_to_numpy=True,
    ).astype("float32")

    n, h = embeddings.shape
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(b"NEGD")
        f.write(struct.pack("<III", 1, n, h))
        for i, t in enumerate(texts):
            tb = t.encode("utf-8")
            f.write(struct.pack("<I", len(tb)))
            f.write(tb)
            f.write(embeddings[i].tobytes())

    print(f"wrote {n} embeddings (dim={h}, normalize={not args.no_normalize}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
