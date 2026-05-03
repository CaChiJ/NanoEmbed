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
