# NanoEmbed

Text embeddings for edge devices, constrained to tens of megabytes of RAM.

See [PLAN.md](PLAN.md) for the milestone-by-milestone build plan.

## Roadmap

- [x] vendoring ggml
- [x] implement GGUF scanner
- [x] implement embedder interface
- [ ] implement layer streaming
- [ ] implement layer-wise batch processing (for performance)
- [ ] implement KV cache compression
- [ ] implement JS wrapper (for node)

## Build

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The public C ABI is in [include/nanoembed/nanoembed.h](include/nanoembed/nanoembed.h)
and is frozen as of M1 — see the milestone matrix in PLAN.md for which
functions are implemented at which point.

## CLI

```sh
echo "hello world" | ./build/bin/nanoembed-cli models/bge-small-en-v1.5-f16.gguf

# inspect a GGUF; --graph also loads it and reports the reserved
# activation buffer (the memory one forward pass can ever need)
./build/bin/nanoembed-inspect models/bge-small-en-v1.5-f16.gguf --graph
```

## Bench

Linux only — the measurement rests on `/proc/<pid>/{status,statm,smaps_rollup,
clear_refs}`, so `nanoembed-bench` is not built on other platforms. The library
and `ctest` still build everywhere.

```sh
# verify the harness itself; needs no model
./build/bin/nanoembed-bench --selftest

# generate baseline / current run (a full suite is ~20 min: 5000 embeds)
.venv/bin/python bench/runner.py --milestone M3.5 --out bench/results/local.json

# compare against the previous milestone baseline
.venv/bin/python bench/compare.py bench/baseline/M3.json bench/results/local.json
```

`nanoembed-bench` runs the workload in a `fork+exec`'d worker and measures it
from the parent, so nothing the harness allocates lands in the numbers. Peak RSS
is scoped to the measurement window by resetting `VmHWM` through `clear_refs`,
and is reported both windowed and lifetime. RSS, PSS and USS are all recorded —
they nearly coincide today and diverge once M4 mmaps the GGUF.

Numbers are machine-specific. Every run stamps an `environment` fingerprint and
`compare.py` refuses (under `--strict`) to compare across machines, so re-run
the baseline on your own hardware before reading any delta.

See `bench/scenarios.yaml` for scenario definitions and PLAN.md §벤치마크
for the metric definitions and regression policy.
