# M3.6 Docker Desktop arm64 comparison baseline

Captured on 2026-08-24 UTC before M4. This is an environment-scoped baseline
for a pre/post-M4 comparison made in the same Docker Desktop VM and mount
stack. It is not a target-hardware baseline and must not be presented as a
general Linux, physical-storage, or Apple Silicon performance result.

## Environment and artifact identity

- Docker Desktop 4.38.0 (Engine 27.5.1), `linux/arm64`.
- Ubuntu image
  `ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517`
  (image ID
  `sha256:5b8c0c14690ed170da4e663fe0bae0d58efe59661e791296ffab28ed2113b650`).
- Ubuntu 24.04.4 LTS, Linux `6.12.5-linuxkit`, aarch64, 8 vCPUs and
  8,217,858,048 bytes guest RAM. The container had no explicit CPU set, CPU
  quota, or memory limit.
- `/src` was the repository bind-mounted read-only. `/build` and `/out` were
  separate writable bind mounts. Docker reported all three as `fakeowner`;
  this filesystem/mount identity is also embedded in every result.
- NanoEmbed Git HEAD:
  `f473b1651e9f9084e6a24f1133061aab5b41032f`, with the intentional A1-A8
  worktree changes present (`dirty: true`). ggml HEAD:
  `387fa29fbbf3149f06a631c7850b6c35c24b0232`.
- Release benchmark binary SHA-256:
  `1e083fda33981e1fbb22a7a1d6c984ce13a1f245347531dc43165023d94b6b6e`.
  It was built with GCC 13.3.0, CMake 3.28.3, `-O3 -DNDEBUG`, shared ggml,
  OpenMP and `GGML_NATIVE=ON`. The complete CMake option fingerprint is in
  every result JSON.
- Scenario SHA-256:
  `19ab58e0115ef6befd60cef8bfaf463e916511f468d4ceb3992bc7c93b82945e`.
  Corpus manifest SHA-256:
  `8d62b41f2f86150191fb46e2e39c3b0ef93352b8e5147e51366d4325fbb97218`.
- Model SHA-256 values: BERT F16
  `f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999`,
  Harrier F32
  `efa662cf610c091dac1fc7378b0f22ad2e4fa7d6510592650c6e405f50b0d78a`,
  and Harrier Q8_0
  `db410d2f5d39bc07f4b69064adfde70f0962da6bb51dc8288c2d1e08e3155b5e`.

`SHA256SUMS` covers all 16 result JSON files, all 16 raw-latency sidecars, the
accuracy report, and the strict self-comparison log. Each result JSON also
contains the size and SHA-256 of its own raw sidecar.

## Measurement matrix

All selections use seed 0. Every mode covers BERT F16 `english_short` and
`long_context`, and Harrier F32/Q8_0 `english_short`, `multilingual_short`, and
`vocab_spread`. Warm/profile-off uses N=3 for BERT `english_short` and N=1 for
the other seven model/group workloads. Every other mode uses N=1 throughout.
The existing scenario iteration counts remain in force: BERT short 50, BERT
long 10, and Harrier 10, with their existing warmup counts. A cold selection
is one fresh native process/worker and one first inference per selected input.

The priority order was completed in full:

1. warm/profile-off;
2. cold/profile-off with `--strict-cold`;
3. warm/profile-on at 10 ms;
4. cold/profile-on at 10 ms with `--strict-cold`.

There are 32 canonical scenario/group results across the four modes. All 16
cold results report `cache_regime: cold`, `cold_cache_verified: true`, and an
eviction-to-worker definition based on zero resident model-file pages observed
by `mincore` after `posix_fadvise(POSIX_FADV_DONTNEED)`. All profile-off results
have zero rollup samples and null PSS/USS. All profile-on results contain
effective rollup samples and non-null RSS/PSS/USS. Profile-on latency remains
diagnostic; profile-off latency/throughput is the comparison value.

## Commands

The container and build were created with:

```sh
docker run -d --name nanoembed-a8-m36-20260824 --platform linux/arm64 \
  --mount type=bind,src=/Users/cachij/Documents/Repos/NanoEmbed,dst=/src,readonly \
  --mount type=bind,src=/private/tmp/nanoembed-a8-build.JucGHh,dst=/build \
  --mount type=bind,src=/private/tmp/nanoembed-a8-output.2i2PmY,dst=/out \
  ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 \
  sleep infinity

docker exec nanoembed-a8-m36-20260824 bash -lc \
  'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
   --no-install-recommends build-essential cmake ninja-build git python3 \
   python3-yaml util-linux ca-certificates && \
   git config --global --add safe.directory /src && \
   git config --global --add safe.directory /src/third_party/ggml && \
   cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=ON \
     -DNANOEMBED_REQUIRE_MODEL=ON && cmake --build /build -j 8'
```

Each artifact was produced by this command shape, with the exact filter/group
rows and mode arguments below. `NAME` is the result basename without `.json`.

```sh
docker exec -w /src nanoembed-a8-m36-20260824 python3 bench/runner.py \
  --milestone M3.6 --filter FILTER GROUP_ARGS --selection-seed 0 MODE_ARGS \
  --bench /build/bin/nanoembed-bench \
  --raw-samples-out /out/NAME.samples.json --out /out/NAME.json
```

| workload | `FILTER` | `GROUP_ARGS` for warm/profile-off | `GROUP_ARGS` for the other modes |
|---|---|---|---|
| BERT short | `single_short_f16` | `--group english_short:3` | `--group english_short:1` |
| BERT long | `single_long_f16` | `--group long_context:1` | `--group long_context:1` |
| Harrier F32 | `single_short_harrier_f32` | `--group english_short:1 --group multilingual_short:1 --group vocab_spread:1` | same |
| Harrier Q8_0 | `single_short_harrier_q8` | `--group english_short:1 --group multilingual_short:1 --group vocab_spread:1` | same |

| mode | `MODE_ARGS` | `NAME` prefix after `M3.6-docker-desktop-arm64-` |
|---|---|---|
| warm/profile-off | `--cache-state warm` | `warm-profile-off-` |
| cold/profile-off | `--cache-state cold --strict-cold` | `cold-profile-off-` |
| warm/profile-on | `--cache-state warm --memory-profile --memory-profile-interval-ms 10` | `warm-profile-on-` |
| cold/profile-on | `--cache-state cold --strict-cold --memory-profile --memory-profile-interval-ms 10` | `cold-profile-on-` |

The accuracy artifact was generated outside benchmark timing:

```sh
docker exec nanoembed-a8-m36-20260824 bash -lc \
  'ctest --test-dir /build -R "^golden$" --output-on-failure && \
   cp /build/accuracy-report.json \
      /out/M3.6-docker-desktop-arm64-accuracy.json'
```

The golden test passed BERT, Harrier F32, Harrier Q8_0, and report-only Harrier
Q4 comparisons. Their fixtures deliberately remain `legacy_unverified`: the
exact upstream Hugging Face revisions were not inferred or fabricated. The
accuracy JSON is therefore useful as the preserved M3.6 implementation/error
result, but not as verified upstream-oracle provenance until fixtures are
regenerated from pinned revisions.

## Limitations

- Docker Desktop's bind mount and storage virtualization sit between the Linux
  guest and macOS storage. Strict cold verification proves guest-visible file
  pages were nonresident; it does not prove every lower host cache was cold.
  Cold latency/I/O is comparable only when M4 uses the same VM and mount stack.
- CPU model, frequency governor and NUMA information were unavailable inside
  the VM. A Docker Desktop restart or resource-setting change invalidates a
  direct comparison even if the container image is unchanged.
- This is one independent run with deliberately small group N, so the stored
  percentiles are descriptive and do not provide a confidence interval or
  broad corpus representativeness.
- The source fingerprint records the commit and dirty state, not a full hash of
  every dirty source file. The preserved Release binary hash is the executable
  identity for this capture; its source tree must not be claimed to equal the
  clean Git commit.
