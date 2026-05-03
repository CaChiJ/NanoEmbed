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
```

## Bench

```sh
# generate baseline / current run
.venv/bin/python bench/runner.py --milestone M3 --out bench/results/local.json

# compare against the milestone baseline
.venv/bin/python bench/compare.py bench/baseline/M3.json bench/results/local.json
```

See `bench/scenarios.yaml` for scenario definitions and PLAN.md §벤치마크
for the regression policy.
