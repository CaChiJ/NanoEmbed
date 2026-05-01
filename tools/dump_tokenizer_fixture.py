#!/usr/bin/env python3
"""Dump HuggingFace tokenizer output for the eval corpus, as a TSV fixture.

The C++ tokenizer test loads this fixture and asserts byte-equal token IDs
for every sentence — that's our oracle for "we tokenize like HuggingFace".

Output format (.tsv):
    # cls=101 sep=102 pad=0 unk=100 max_seq_len=512 model=...
    <text>\t<id1>,<id2>,...,<idN>
    <text>\t<id1>,<id2>,...,<idN>
    ...

The TSV format is dead-simple to parse from C++ (no JSON dep) at the cost
of disallowing tabs/newlines inside text — verified at write time below.

Re-run only if the corpus or target model changes.
"""

import argparse
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--corpus", type=pathlib.Path,
                        default=pathlib.Path("tests/corpus/eval.txt"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("tests/fixtures/tokenizer/bge-small-eval.tsv"))
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("error: pip install -r requirements-dev.txt", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(args.model)

    with args.corpus.open(encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            f"# cls={tok.cls_token_id} sep={tok.sep_token_id} "
            f"pad={tok.pad_token_id} unk={tok.unk_token_id} "
            f"max_seq_len={args.max_length} "
            f"do_lower_case={int(getattr(tok, 'do_lower_case', True))} "
            f"n={len(texts)} model={args.model}\n"
        )
        n_total = 0
        for t in texts:
            if "\t" in t or "\n" in t:
                raise SystemExit(f"corpus has tab/newline in text: {t!r}")
            enc = tok(t, padding=False, truncation=True,
                      max_length=args.max_length, add_special_tokens=True)
            ids = enc["input_ids"]
            f.write(t + "\t" + ",".join(str(i) for i in ids) + "\n")
            n_total += len(ids)

    print(f"wrote {len(texts)} samples ({n_total} tokens) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
