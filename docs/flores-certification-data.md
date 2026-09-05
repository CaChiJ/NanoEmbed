# FLORES+ accuracy certification data

This data path is for release certification, not the fast developer `golden`
CTest. It prepares every public `dev` and `devtest` record from a pinned
FLORES+ snapshot as deterministic local JSONL shards.

## Source contract

The committed source specification is
`tests/certification/flores_plus_v4_6.source.json`. It pins:

- the Hugging Face dataset ID and exact 40-character commit;
- the FLORES+ release version and license;
- the included splits and derived shard size; and
- the access page whose conditions must be accepted.

Changing the dataset revision is a reviewable certification-input change. Do
not replace it with `main` or another mutable revision.

## Prepare the local corpus

FLORES+ is gated to protect evaluation integrity. The person operating the
home server must first accept the conditions at
<https://huggingface.co/datasets/openlanguagedata/flores_plus> and authenticate:

```sh
hf auth login
python3 tools/prepare_flores_accuracy_corpus.py
```

The command downloads only the pinned `dev`/`devtest` JSONL files and source
metadata. It then writes `certification-data/flores-plus-v4.6/` containing:

- canonical JSONL shards in deterministic split/file/line order;
- `manifest.json` with every source and shard SHA-256;
- `manifest.sha256` anchoring the manifest itself;
- sample counts by split, language variety, script, and domain; and
- the generator command, script hash, Python version, and platform.

An unauthorized, incomplete, malformed, duplicate, empty, or wrong-revision
dataset fails the command. It never falls back to a smaller corpus.

Verify an existing corpus without network access:

```sh
python3 tools/prepare_flores_accuracy_corpus.py --verify-only
```

The generated directory is ignored by Git. FLORES+ is CC BY-SA 4.0 and its
dataset card asks that downloaded data not be publicly redistributed without
protection from automated scraping. Keep the text and future multi-gigabyte
golden shards on the certification server; publish only an approved manifest
and aggregate certification report.

## Development test

The preparation logic has a synthetic unit test and does not require FLORES+
credentials:

```sh
ctest --test-dir build -R '^flores_accuracy_corpus$' --output-on-failure
```

The next certification stage consumes these canonical shards to generate
sharded PyTorch reference embeddings and compare NanoEmbed output. That stage
must be validated against a real prepared manifest before its release gate is
enabled.
