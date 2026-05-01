#!/usr/bin/env python3
"""Dump per-layer activations from HuggingFace BERT for regression fixtures.

For each input text we run BAAI/bge-small-en-v1.5 in float32 and capture:
  - input_ids, attention_mask
  - embed_out          : [1, S, H]  output of the embedding stage
                          (token + position + token-type + LN)
  - layer_0_out ... layer_11_out : [1, S, H]  output of each encoder block

Output format ("NEMB" multi-tensor binary, see _write_tensors below):
    "NEMB" (4 bytes)
    version (u32, =1)
    n_tensors (u32)
    For each:
        name_len (u32) + name (utf-8)
        dtype (u32: 0=F32, 1=I32, 2=I64)
        n_dims (u32) + dims (i64[n_dims])
        data (raw bytes)

The C++ forward parity test parses this format directly — no .npy
parsing dependency. All numbers are little-endian.
"""

import argparse
import pathlib
import struct
import sys
from typing import Dict


DTYPE_F32 = 0
DTYPE_I32 = 1
DTYPE_I64 = 2


def _dtype_code(arr) -> int:
    import numpy as np
    if arr.dtype == np.float32: return DTYPE_F32
    if arr.dtype == np.int32:   return DTYPE_I32
    if arr.dtype == np.int64:   return DTYPE_I64
    raise ValueError(f"unsupported dtype: {arr.dtype}")


def _write_tensors(path: pathlib.Path, tensors: Dict[str, "np.ndarray"]) -> None:
    with path.open("wb") as f:
        f.write(b"NEMB")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, arr in tensors.items():
            n = name.encode("utf-8")
            f.write(struct.pack("<I", len(n)))
            f.write(n)
            f.write(struct.pack("<II", _dtype_code(arr), arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<q", int(d)))
            f.write(arr.tobytes())


def dump_one(model, tokenizer, text: str):
    import numpy as np
    import torch

    enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True)

    captures: Dict[str, "np.ndarray"] = {}
    hooks = []

    def grab(name):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            captures[name] = t.detach().to(torch.float32).cpu().numpy()
        return hook

    hooks.append(model.embeddings.register_forward_hook(grab("embed_out")))
    for i, layer in enumerate(model.encoder.layer):
        hooks.append(layer.register_forward_hook(grab(f"layer_{i}_out")))

    with torch.no_grad():
        model(**enc)

    for h in hooks:
        h.remove()

    captures["input_ids"]      = enc["input_ids"].numpy().astype(np.int32)
    captures["attention_mask"] = enc["attention_mask"].numpy().astype(np.int32)
    return captures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path("tests/fixtures/activations"))
    parser.add_argument("--inputs", nargs="*", default=[
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "A man is cutting carpet with a knife.",
    ])
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("error: pip install -r requirements-dev.txt (need torch + transformers)",
              file=sys.stderr)
        return 2

    model = AutoModel.from_pretrained(args.model)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx, text in enumerate(args.inputs):
        captures = dump_one(model, tok, text)
        slug = f"sample_{idx:02d}"
        out = args.out_dir / f"{slug}.nemb"
        _write_tensors(out, captures)
        # Companion .txt with the input text for human readability.
        (args.out_dir / f"{slug}.txt").write_text(text + "\n", encoding="utf-8")
        n_tok = captures["input_ids"].shape[1]
        print(f"wrote {out} ({n_tok} tokens, {len(captures)} tensors)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
