#!/usr/bin/env python3
"""Dump HuggingFace tokenizer output for the eval corpus, as a TSV fixture.

The C++ tokenizer test loads this fixture and asserts byte-equal token IDs
for every sentence — that's our oracle for "we tokenize like HuggingFace".

Output format (.tsv):
    # key=value key=value ...
    <text>\t<id1>,<id2>,...,<idN>
    <text>\t<id1>,<id2>,...,<idN>
    ...

The header carries whatever the tokenizer states about itself, with -1 for
"this family has no such token". The C++ test asserts against those values
rather than hardcoding one model's constants, which is what lets a single
test binary cover both WordPiece and SentencePiece-BPE models.

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
    parser.add_argument("--corpus", type=pathlib.Path, nargs="+",
                        default=[pathlib.Path("tests/corpus/eval.txt")],
                        help="one or more corpus files, concatenated in order")
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

    texts = []
    for corpus in args.corpus:
        with corpus.open(encoding="utf-8") as f:
            texts += [line.rstrip("\n") for line in f if line.strip()]

    def tid(name):
        # -1 means the family has no such token, which C++ can parse; "None"
        # cannot.
        v = getattr(tok, name, None)
        return -1 if v is None else v

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            f"# model={args.model} n={len(texts)} "
            # Base vocab, not len(tok): the latter counts added tokens
            # a text-only GGUF export does not carry (harrier adds a
            # multimodal <image_soft_token> past the end of the vocab).
            f"vocab_size={tok.vocab_size} max_seq_len={args.max_length} "
            f"cls={tid('cls_token_id')} sep={tid('sep_token_id')} "
            f"pad={tid('pad_token_id')} unk={tid('unk_token_id')} "
            f"bos={tid('bos_token_id')} eos={tid('eos_token_id')} "
            f"do_lower_case={int(getattr(tok, 'do_lower_case', False))}\n"
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
