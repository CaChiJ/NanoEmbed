# M4 Docker Desktop arm64 measurement environment

This bundle was collected in the same still-running A8 container and mount stack documented by `bench/baseline/M3.6-docker-desktop-arm64/ENVIRONMENT.md`. It is valid only as a same-Docker-Desktop-VM comparison.

- Docker Desktop 4.38.0 (Engine 27.5.1), Ubuntu 24.04.4 LTS `linux/arm64`, Linux `6.12.5-linuxkit`, 8 vCPUs, 8,217,858,048 guest bytes.
- Image: `ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517` (image ID `sha256:5b8c0c14690ed170da4e663fe0bae0d58efe59661e791296ffab28ed2113b650`).
- Container: `nanoembed-a8-m36-20260824`.
- `/src`: repository bind mount, read-only, `fakeowner`.
- `/build`: fixed writable bind mount `/private/tmp/nanoembed-a8-build.JucGHh`, `fakeowner`.
- `/out`: fixed writable bind mount `/private/tmp/nanoembed-a8-output.2i2PmY`, `fakeowner`.
- Git HEAD: `f473b1651e9f9084e6a24f1133061aab5b41032f`; intentional dirty A1-B4 worktree. ggml: `387fa29fbb3149f06a631c7850b6c35c24b0232`.
- Clean-first Release rebuild: `docker exec nanoembed-a8-m36-20260824 bash -lc 'cmake --build /build --clean-first -j 8'`.
- Frozen M4 benchmark binary SHA-256: `8abd7e2840fc80317ff119bd175e6c3e8eb1be2355326d2cc913490f7afb481c`, GCC 13.3.0, `-O3 -DNDEBUG`, shared ggml, OpenMP, `GGML_NATIVE=ON`.
- Scenario SHA-256: `85aab11134476c24b5b9b673c6d9cfcb44821b20157a34508e375c139d6bb261`; it differs from M3.6 because M4 adds streaming/Q4 definitions. Row-level controlled settings are validated.
- Corpus manifest SHA-256: `8d62b41f2f86150191fb46e2e39c3b0ef93352b8e5147e51366d4325fbb97218`, seed 0.

## Matrix

For eager and streaming separately, the runner collected warm/profile-off, cold/profile-off (`--strict-cold`), warm/profile-on (10 ms), and cold/profile-on (10 ms plus `--strict-cold`). Baseline-overlap workloads are BERT F16 short/long and Harrier F32/Q8_0 `english_short`, `multilingual_short`, and `vocab_spread`. Q4_K is M4-only report-only `english_short`.

Warm/profile-off BERT short uses N=3, matching M3.6. Every other selection uses N=1. Scenario warmup/iteration settings were not overridden. There are 40 result files and 40 sidecars containing 72 workload rows. Together with accuracy, machine comparison, environment, report and validation evidence, the bundle has 92 checksummed payload files plus `SHA256SUMS`.

All exact invocations are in `measurement-commands.log`. Accuracy was run separately with:

```sh
docker exec nanoembed-a8-m36-20260824 bash -lc   'ctest --test-dir /build -R "^golden$" --output-on-failure'
```

It passed in 146.87 seconds and `/build/accuracy-report.json` was copied into this bundle before performance measurement began.

## Validation commands

```sh
cd bench/baseline/M3.6-docker-desktop-arm64
shasum -a 256 -c SHA256SUMS

python3 bench/compare.py BASELINE_RESULT M4_EAGER_MATCHING_RESULT

cd bench/results/M4-docker-desktop-arm64
shasum -a 256 -c SHA256SUMS
```

`baseline-validation.log`, `validation.log`, `comparison-validator.log`, and `binary-environment-freeze.log` preserve the outputs. The current binary hash was checked before every measurement phase and once after all 40 runs.

## Scope limits

Docker Desktop bind mounts and host caches remain below guest-visible `fadvise`/`mincore`. CPU frequency policy, CPU model, and NUMA were unavailable. Results use one independent run and no confidence interval. See `CLOSEOUT.md` for memory-sampling, advisory-residency, activation, token-table, accuracy-provenance, and Q4 limitations.
