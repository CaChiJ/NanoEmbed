# M4 Docker Desktop arm64 closeout

This report closes M4 only for the preserved Docker Desktop 4.38.0 Ubuntu 24.04 `linux/arm64` VM/container and the exact `/src` and `/build` bind mounts used by the M3.6 baseline. It is not a physical-target or general-Linux performance claim.

## Outcome

- The full matrix completed: 40 bounded result JSON files, 40 raw-latency sidecars, 72 M4 workload rows, 36 same-binary eager/streaming pairs, 32 M3.6 overlap rows, and 8 accuracy comparisons. The bundle has 93 files total: 92 payload files covered by `SHA256SUMS`, plus the checksum manifest itself.
- All 1,798 result/schema/hash/sample checks plus 15 environment/mount checks passed; all 36 cold rows were strict-cache verified.
- Streaming and eager outputs were identical for all four accuracy families in the saved full-vector comparison: worst cosine was effectively 1.0 and maximum/mean absolute error, RMSE, and norm difference were zero. Existing PyTorch cosine gates passed unchanged.
- Streaming sharply reduced observed process memory, but its steady-state speed cost depended on model/quantization and corpus. The same-binary warm/profile-off rows below are the authoritative feature comparison.
- M4 eager differed materially from M3.6 in several Q8 warm and cold-start rows. That drift is visible rather than normalized away; one independent run cannot separate code/lifecycle effects from run variance.

Profile-off latency/throughput is authoritative. Profile-on latency is diagnostic and is never mixed into the performance tables. Percent changes are `(streaming - eager) / eager`; lower latency/RSS and higher throughput are favorable.

## Same-binary M4 feature effect: warm/profile-off

| workload | p50 ms E→S | p90 ms E→S | p95 ms E→S | p99 ms E→S | items/s E→S | RSS lifetime MiB E→S |
|---|---:|---:|---:|---:|---:|---:|
| BERT F16 long | 2561.59→2558.09 (-0.1%) | 2580.37→2588.66 (+0.3%) | 2580.37→2588.66 (+0.3%) | 2580.37→2588.66 (+0.3%) | 0.39→0.39 (+0.1%) | 91.94→34.02 (-63.0%) |
| BERT F16 short | 35.96→37.50 (+4.3%) | 50.20→50.81 (+1.2%) | 50.91→51.77 (+1.7%) | 61.24→61.62 (+0.6%) | 24.41→23.59 (-3.4%) | 79.07→14.90 (-81.2%) |
| Harrier F32 | 310.65→315.61 (+1.6%) | 318.66→364.36 (+14.3%) | 318.66→364.36 (+14.3%) | 318.66→364.36 (+14.3%) | 3.19→3.04 (-4.5%) | 1112.01→102.24 (-90.8%) |
| Harrier F32 / multilingual_short | 151.85→165.28 (+8.8%) | 154.38→169.44 (+9.8%) | 154.38→169.44 (+9.8%) | 154.38→169.44 (+9.8%) | 6.59→6.01 (-8.8%) | 1113.90→102.32 (-90.8%) |
| Harrier F32 / vocab_spread | 617.75→642.65 (+4.0%) | 665.26→1129.04 (+69.7%) | 665.26→1129.04 (+69.7%) | 665.26→1129.04 (+69.7%) | 1.54→1.28 (-17.2%) | 1111.97→102.94 (-90.7%) |
| Harrier Q4_K | 34.98→40.30 (+15.2%) | 35.49→43.36 (+22.2%) | 35.49→43.36 (+22.2%) | 35.49→43.36 (+22.2%) | 28.36→24.39 (-14.0%) | 313.26→84.62 (-73.0%) |
| Harrier Q8_0 | 13.62→17.48 (+28.4%) | 14.68→17.94 (+22.2%) | 14.68→17.94 (+22.2%) | 14.68→17.94 (+22.2%) | 72.73→56.93 (-21.7%) | 360.85→87.04 (-75.9%) |
| Harrier Q8_0 / multilingual_short | 7.61→13.26 (+74.3%) | 9.83→14.31 (+45.5%) | 9.83→14.31 (+45.5%) | 9.83→14.31 (+45.5%) | 113.44→73.29 (-35.4%) | 363.04→87.04 (-76.0%) |
| Harrier Q8_0 / vocab_spread | 26.45→36.90 (+39.5%) | 28.11→56.74 (+101.8%) | 28.11→56.74 (+101.8%) | 28.11→56.74 (+101.8%) | 37.43→23.19 (-38.0%) | 360.83→87.05 (-75.9%) |

BERT long was essentially throughput-neutral while its RSS fell 63.0%; BERT short lost 3.4% throughput while RSS fell 81.2%. Harrier F32 RSS fell about 90.7–90.8% with throughput changes from -4.5% to -17.2%. Harrier Q8_0 RSS fell about 75.9–76.0%, while throughput fell 21.7–38.0%. Q4_K is report-only: RSS fell 73.0% and throughput fell 14.0%. The F32 `vocab_spread` and Q8 tails show outliers in only ten timed observations, so the percentiles are descriptive.

## Same-binary M4 feature effect: cold/profile-off

Each row has one independently strict-verified cold worker/sample; all percentiles of that row are therefore the same observation. Canonical cold cost is `startup_to_first_result_ms`, not the internal model-load/context-create split.

| workload | first inference p50 ms E→S | startup-to-result p50 ms E→S | items/s E→S | RSS lifetime MiB E→S | major faults E→S | minor faults E→S | I/O read bytes E→S |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT F16 long | 2555.37→2547.42 (-0.3%) | 2604.09→2559.72 (-1.7%) | 0.38→0.39 (+1.7%) | 92.25→33.80 (-63.4%) | 11→120 | 5448→6609 | 0→0 |
| BERT F16 short | 37.62→84.86 (+125.6%) | 105.67→98.28 (-7.0%) | 9.46→10.18 (+7.5%) | 78.87→14.67 (-81.4%) | 11→118 | 3973→3953 | 0→0 |
| Harrier F32 | 315.83→450.18 (+42.5%) | 1682.16→863.25 (-48.7%) | 0.59→1.16 (+94.9%) | 1111.94→102.31 (-90.8%) | 12→365 | 18498→23379 | 0→0 |
| Harrier F32 / multilingual_short | 158.55→367.62 (+131.9%) | 1600.73→1012.36 (-36.8%) | 0.62→0.99 (+58.1%) | 1111.96→102.20 (-90.8%) | 12→496 | 18495→23325 | 0→0 |
| Harrier F32 / vocab_spread | 612.03→1127.25 (+84.2%) | 2379.80→1595.34 (-33.0%) | 0.42→0.63 (+49.2%) | 1111.84→102.69 (-90.8%) | 10→545 | 16464→23178 | 0→0 |
| Harrier Q4_K | 36.33→98.67 (+171.6%) | 631.98→549.77 (-13.0%) | 1.58→1.82 (+15.0%) | 315.24→84.46 (-73.2%) | 14→235 | 17424→14895 | 0→0 |
| Harrier Q8_0 | 15.28→140.05 (+816.4%) | 689.27→600.31 (-12.9%) | 1.45→1.67 (+14.8%) | 361.09→86.49 (-76.0%) | 14→244 | 18933→17796 | 0→0 |
| Harrier Q8_0 / multilingual_short | 14.80→75.21 (+408.1%) | 670.35→524.88 (-21.7%) | 1.49→1.91 (+27.7%) | 361.03→87.12 (-75.9%) | 15→257 | 17945→17291 | 0→0 |
| Harrier Q8_0 / vocab_spread | 38.47→95.82 (+149.1%) | 711.39→596.44 (-16.2%) | 1.41→1.68 (+19.3%) | 361.03→86.96 (-75.9%) | 15→273 | 18440→17879 | 0→0 |

Streaming reduced canonical cold startup by 12.9–48.7% for all Harrier rows and 7.0% for BERT short; BERT long changed by -1.7%. First-inference-only latency often increased because page faults moved into phase execution, while the eliminated eager whole-weight preparation dominated total startup. Major/minor fault counts increased in streaming. Linux reported zero `io_read_bytes` for these Docker runs; that does not establish that no lower host/storage I/O occurred.

## Profile-on sampled memory (diagnostic runs)

PSS/USS are 10 ms sampled peaks and therefore lower bounds, not kernel high-water marks. The sampling thread performs `smaps_rollup` page-table walks and can perturb both timing and residency.

### Warm profile-on

| workload | RSS lifetime MiB E→S | PSS sampled peak MiB E→S | USS sampled peak MiB E→S | samples E/S |
|---|---:|---:|---:|---:|
| BERT F16 long | 91.93→33.97 (-63.0%) | 90.40→32.08 (-64.5%) | 88.98→30.66 (-65.5%) | 2085/2091 |
| BERT F16 short | 79.12→14.88 (-81.2%) | 77.13→12.80 (-83.4%) | 75.71→11.38 (-85.0%) | 153/156 |
| Harrier F32 | 1118.00→102.39 (-90.8%) | 1115.92→100.38 (-91.0%) | 1114.50→98.96 (-91.1%) | 267/260 |
| Harrier F32 / multilingual_short | 1111.71→102.43 (-90.8%) | 1109.92→100.35 (-91.0%) | 1108.50→98.93 (-91.1%) | 128/142 |
| Harrier F32 / vocab_spread | 1111.82→102.56 (-90.8%) | 1109.95→100.53 (-90.9%) | 1108.53→99.11 (-91.1%) | 511/511 |
| Harrier Q4_K | 313.27→84.63 (-73.0%) | 311.27→82.49 (-73.5%) | 309.85→81.07 (-73.8%) | 40/36 |
| Harrier Q8_0 | 362.83→86.89 (-76.1%) | 361.08→84.92 (-76.5%) | 359.66→83.50 (-76.8%) | 17/40 |
| Harrier Q8_0 / multilingual_short | 360.98→87.05 (-75.9%) | 359.09→84.92 (-76.4%) | 357.67→83.50 (-76.7%) | 8/13 |
| Harrier Q8_0 / vocab_spread | 362.99→87.30 (-75.9%) | 361.11→85.32 (-76.4%) | 359.69→83.90 (-76.7%) | 22/35 |

### Cold profile-on

| workload | RSS lifetime MiB E→S | PSS sampled peak MiB E→S | USS sampled peak MiB E→S | samples E/S |
|---|---:|---:|---:|---:|
| BERT F16 long | 92.34→33.57 (-63.6%) | 90.39→31.85 (-64.8%) | 88.97→30.43 (-65.8%) | 204/212 |
| BERT F16 short | 79.02→14.70 (-81.4%) | 77.13→12.16 (-84.2%) | 75.71→10.74 (-85.8%) | 10/7 |
| Harrier F32 | 1111.86→102.37 (-90.8%) | 1109.92→99.34 (-91.0%) | 1108.50→97.92 (-91.2%) | 172/72 |
| Harrier F32 / multilingual_short | 1111.97→102.22 (-90.8%) | 1109.91→99.72 (-91.0%) | 1108.49→98.30 (-91.1%) | 105/69 |
| Harrier F32 / vocab_spread | 1112.02→102.44 (-90.8%) | 1109.99→100.33 (-91.0%) | 1108.57→98.91 (-91.1%) | 281/115 |
| Harrier Q4_K | 313.28→84.38 (-73.1%) | 311.29→81.12 (-73.9%) | 309.88→79.71 (-74.3%) | 60/45 |
| Harrier Q8_0 | 363.06→86.98 (-76.0%) | 361.08→83.67 (-76.8%) | 359.66→82.25 (-77.1%) | 64/45 |
| Harrier Q8_0 / multilingual_short | 363.20→86.97 (-76.1%) | 361.07→84.48 (-76.6%) | 359.65→83.06 (-76.9%) | 52/47 |
| Harrier Q8_0 / vocab_spread | 361.07→87.30 (-75.8%) | 359.11→85.17 (-76.3%) | 357.69→83.75 (-76.6%) | 65/69 |

## M4 eager versus M3.6 drift check

These are not the streaming feature effect. They expose harness/lifecycle/code drift using matching model, corpus selection/hash, pooling, normalization, threads, warmup/iteration, cache, and profile settings. M4 metadata-only load moves eager preparation into context creation, so individual `model_load_ms` and `context_create_ms` are not interpreted as regressions.

### Warm/profile-off

| workload | p50 ms M3.6→M4 | p99 ms M3.6→M4 | items/s M3.6→M4 | RSS MiB M3.6→M4 |
|---|---:|---:|---:|---:|
| BERT F16 long | 2531.12→2561.59 (+1.2%) | 2552.99→2580.37 (+1.1%) | 0.39→0.39 (-0.9%) | 92.11→91.94 (-0.2%) |
| BERT F16 short | 34.86→35.96 (+3.2%) | 63.34→61.24 (-3.3%) | 24.54→24.41 (-0.5%) | 79.00→79.07 (+0.1%) |
| Harrier F32 | 302.68→310.65 (+2.6%) | 306.56→318.66 (+3.9%) | 3.29→3.19 (-3.1%) | 1110.35→1112.01 (+0.1%) |
| Harrier F32 / multilingual_short | 151.02→151.85 (+0.5%) | 154.44→154.38 (-0.0%) | 6.58→6.59 (+0.0%) | 1110.31→1113.90 (+0.3%) |
| Harrier F32 / vocab_spread | 603.76→617.75 (+2.3%) | 612.23→665.26 (+8.7%) | 1.65→1.54 (-6.4%) | 1110.32→1111.97 (+0.1%) |
| Harrier Q8_0 | 12.27→13.62 (+11.0%) | 12.29→14.68 (+19.5%) | 81.49→72.73 (-10.8%) | 359.60→360.85 (+0.3%) |
| Harrier Q8_0 / multilingual_short | 7.02→7.61 (+8.4%) | 7.04→9.83 (+39.6%) | 142.57→113.44 (-20.4%) | 359.27→363.04 (+1.0%) |
| Harrier Q8_0 / vocab_spread | 26.64→26.45 (-0.7%) | 27.06→28.11 (+3.9%) | 36.43→37.43 (+2.7%) | 359.43→360.83 (+0.4%) |

### Cold/profile-off

| workload | first inference ms M3.6→M4 | startup-to-result ms M3.6→M4 | items/s M3.6→M4 | RSS MiB M3.6→M4 |
|---|---:|---:|---:|---:|
| BERT F16 long | 2534.03→2555.37 (+0.8%) | 2602.79→2604.09 (+0.0%) | 0.38→0.38 (-0.0%) | 91.86→92.25 (+0.4%) |
| BERT F16 short | 35.90→37.62 (+4.8%) | 83.99→105.67 (+25.8%) | 11.91→9.46 (-20.5%) | 78.99→78.87 (-0.2%) |
| Harrier F32 | 307.55→315.83 (+2.7%) | 1582.11→1682.16 (+6.3%) | 0.63→0.59 (-5.9%) | 1110.20→1111.94 (+0.2%) |
| Harrier F32 / multilingual_short | 152.52→158.55 (+4.0%) | 1056.25→1600.73 (+51.5%) | 0.95→0.62 (-34.0%) | 1110.17→1111.96 (+0.2%) |
| Harrier F32 / vocab_spread | 609.47→612.03 (+0.4%) | 1461.45→2379.80 (+62.8%) | 0.68→0.42 (-38.6%) | 1110.18→1111.84 (+0.1%) |
| Harrier Q8_0 | 17.47→15.28 (-12.5%) | 646.29→689.27 (+6.7%) | 1.55→1.45 (-6.2%) | 361.61→361.09 (-0.1%) |
| Harrier Q8_0 / multilingual_short | 11.02→14.80 (+34.3%) | 579.42→670.35 (+15.7%) | 1.73→1.49 (-13.6%) | 359.44→361.03 (+0.4%) |
| Harrier Q8_0 / vocab_spread | 47.66→38.47 (-19.3%) | 622.40→711.39 (+14.3%) | 1.61→1.41 (-12.5%) | 359.45→361.03 (+0.4%) |

Warm eager RSS remained within about 1.1% of M3.6. Warm Q8_0 throughput drifted from +2.7% to -20.4% depending on group; other rows ranged from +0.03% to -6.38%. Cold startup drift ranged from +0.05% to +62.8%, with the largest changes on Harrier F32 multilingual/vocab rows. Because `independent_runs=1`, these differences are diagnostics and no confidence interval or statistical significance is claimed.

## Accuracy and provenance

Accuracy was generated by `golden` outside every performance run. The PyTorch fixtures are preserved but explicitly `legacy_unverified`: exact upstream revisions and original generator environments were not recorded and were not fabricated.

### Streaming versus PyTorch

| precision | worst cosine | worst max abs | mean mean-abs | mean RMSE | mean norm diff | gate |
|---|---:|---:|---:|---:|---:|---|
| f16_weights_f32_compute | 0.999999156 | 0.000235654414 | 4.17022861e-05 | 5.22410139e-05 | 4.12822208e-08 | passed; 0 failed |
| f32_weights_f32_compute | 0.999999725 | 0.000107072294 | 7.22054406e-06 | 9.1647673e-06 | 3.93523806e-08 | passed; 0 failed |
| q8_0_weights_f32_compute | 0.999105757 | 0.00678130006 | 0.000674837495 | 0.000852970877 | 4.30002193e-08 | passed; 0 failed |
| q4_k_weights_f32_compute (report-only) | 0.947608256 | 0.0471795155 | 0.00539896527 | 0.00682612682 | 4.06011606e-08 | passed; 0 failed |

### Streaming versus eager full-vector error

| precision | worst cosine | worst max abs | worst mean abs | worst RMSE | worst norm diff |
|---|---:|---:|---:|---:|---:|
| f16_weights_f32_compute | 0.99999999999999978 | 0 | 0 | 0 | 0 |
| f32_weights_f32_compute | 0.99999999999999978 | 0 | 0 | 0 | 0 |
| q8_0_weights_f32_compute | 0.99999999999999978 | 0 | 0 | 0 | 0 |
| q4_k_weights_f32_compute (report-only) | 0.99999999999999978 | 0 | 0 | 0 | 0 |

## Interpretation limits

- Scope is the same running Docker Desktop arm64 VM/container and bind-mount stack only. Guest-visible strict cold verification does not evict or prove the state of every lower macOS/Docker cache.
- `posix_fadvise` and `madvise` are advisory. `mincore` reports guest-visible mapping residency; it is evidence for the stated cache boundary, not physical-storage behavior.
- M4 streams transformer layer ranges but still requires the token embedding table. Harrier’s table is the dominant immutable range, so streaming cannot make model memory approach activation-only size.
- Activations use a context-local F32 ping-pong allocation; that floor remains even when weight pages are advised away.
- PSS/USS peaks are 10 ms samples and lower bounds. Profile-on latency is diagnostic because `smaps_rollup` is an observer.
- Streaming increases page faults by design. Docker `io_read_bytes=0` is not evidence of zero lower-level reads.
- Q4_K has no M3.6 performance baseline and remains report-only.
- All comparisons are one independent run with null CI. Percentiles, especially cold N=1 and warm Harrier N=10, are descriptive.
- Accuracy provenance remains `legacy_unverified` until fixtures are regenerated from pinned upstream revisions.

## Saved evidence

- `comparison-summary.json`: all 32 baseline-drift rows and all 36 same-binary feature pairs, including p50/p90/p95/p99, throughput, RSS/PSS/USS, faults, I/O, and cold startup deltas.
- `validation.log`: per-file schema, strict execution-mode evidence, fingerprint/hash, selection, sidecar, sample, cache, and memory-profile checks.
- `comparison-validator.log`: unchanged `bench/compare.py` output for all 16 overlapping result-file pairs.
- `measurement-commands.log`: all 40 exact runner commands.
- `accuracy-validation.log` and accuracy JSON: separate golden execution and full error/provenance output.
- `SHA256SUMS`: integrity for every file in this bundle except the manifest itself.

---

# Addendum: post-M4 layer partitioning (2026-08-26)

This section is not part of the M4 matrix above. It records a later change that
cuts a transformer block into several graphs instead of one, and reports what
that does to leased weight bytes. It is a different instrument in a different
container, and it measures a different quantity, so none of the M4 numbers above
are restated, revised, or superseded by it.

The default remains one graph per block. Every table above therefore continues
to describe the shipped configuration exactly as measured.

## What changed

A layer is now declared as a sequence of streaming units, each holding both the
GGUF tensor names it needs resident and the graph nodes that consume them. A
partition preset merges contiguous runs of that sequence into groups, and each
group takes one lease and runs as one graph. Presets are selected by
`NANOEMBED_STREAMING_PARTITION` and are internal: the frozen C ABI is unchanged
and `use_streaming=1` still resolves to the `layer` preset.

- `layer` — one group per block. The M4 behavior, and the default.
- `attn-ffn` — cut where the attention and FFN halves meet.
- `budget:N` — greedy, extend a run while its weights stay within N bytes.
- `unit` — no merging.

## Measurement instrument and scope

Two instruments are used below, and they answer different questions.

The **leased weight bytes** table comes from `nanoembed_streaming_integration_test`'s
own residency diagnostics, not from `nanoembed-bench`. `peak group KiB` is the
largest group's summed page-aligned tensor bytes — a static property of the
model and preset. `advised peak KiB` is the high-water of bytes simultaneously
under an active lease plus the lifetime-retained common ranges, sampled while
only one context had run. **These are bytes the code asked the kernel to keep,
not observed residency** — `madvise` is advisory, so this counter says what was
requested and nothing about what the kernel did.

The **measured RSS/PSS/USS** section further down comes from `nanoembed-bench`
itself, sampling `/proc/<pid>/smaps_rollup` the same way the M4 matrix's
profile-on rows did. That section states its own corpus and interval; this
paragraph covers what both share.

Both ran in a plain `ubuntu:24.04` `linux/arm64` container with
`build-essential`/`cmake`/`ninja`/`python3`/`pyyaml`, built for this work. It is
**not** the preserved Docker Desktop 4.38.0 container or the `/src` and `/build`
bind mounts the M4 matrix required, so absolute figures below are not to be
read against the M4 tables above line-for-line even where the measurement class
matches.

## Leased weight bytes by preset

Serial context, 8 samples per model.

### harrier-270m F32 — one block is 21,772 KiB

| preset | groups/layer | peak group KiB | vs `layer` | advised peak KiB | willneed | dontneed | gallocr replans | slot KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `layer` | 1 | 21772 | — | 21782 | 382 | 330 | 24 | 77 |
| `attn-ffn` | 2 | 15365 | -29.4% | 15374 | 598 | 510 | 304 | 155 |
| `budget:10MiB` | 3 | 10240 | -53.0% | 10250 | 1030 | 875 | 448 | 480 |
| `unit` | 8 | 10240 | -53.0% | 10250 | 1894 | 1692 | 1160 | 899 |

### harrier-270m Q8_0 — one block is 5,792 KiB

| preset | groups/layer | peak group KiB | vs `layer` | advised peak KiB | willneed | dontneed | gallocr replans | slot KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `layer` | 1 | 5792 | — | 5802 | 370 | 321 | 24 | 77 |
| `attn-ffn` | 2 | 4085 | -29.5% | 4094 | 586 | 547 | 304 | 155 |
| `budget:10MiB` | 1 | 5792 | 0.0% | 5802 | 370 | 319 | 24 | 77 |
| `unit` | 8 | 2720 | -53.0% | 2730 | 1882 | 1775 | 1160 | 899 |

### bge-small-en-v1.5 F16 — one block is 3,475 KiB

| preset | groups/layer | peak group KiB | vs `layer` | advised peak KiB | willneed | dontneed | gallocr replans | slot KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `layer` | 1 | 3475 | — | 3876 | 281 | 244 | 24 | 40 |
| `attn-ffn` | 2 | 2314 | -33.4% | 2716 | 425 | 365 | 208 | 81 |
| `budget:10MiB` | 1 | 3475 | 0.0% | 3876 | 281 | 244 | 24 | 40 |
| `unit` | 4 | 1158 | -66.7% | 1560 | 713 | 604 | 312 | 364 |

A whole bge-small or Q8_0 block fits inside a 10 MiB budget, so `budget:10MiB`
leaves both unmerged at one group and is identical to `layer` for them. BERT's
advised peak sits about 400 KiB above its peak group because its common class
carries the learned position table; gemma3's common class is one norm vector.

## Accuracy

Every preset on every model reported a worst streaming-versus-eager cosine of
1.000000000 across 8 samples. The existing PyTorch cosine gates passed unchanged
under all four presets. This matches the M4 finding that streaming and eager
outputs are identical, and extends it to each partition.

## Measured RSS/PSS/USS peaks

The leased-byte table above says what the code asked the kernel to keep. This
section says what `nanoembed-bench` actually observed, using the same
`/proc/<pid>/smaps_rollup` sampler and the same 10 ms interval the M4 matrix
used (`metrics.cpp`, `--memory-profile --memory-profile-interval-ms 10`).

**Not directly comparable to the M4 matrix's Harrier F32 rows.** The container
is still the plain `ubuntu:24.04` image built for this addendum, not the
preserved Docker Desktop bundle, and the corpus differs: this run iterates 10
times over the full 100-sentence `english_short` group (1,000 timed requests,
500 warmup requests) rather than the matrix's fixed 3-sentence cross-group
sample. It is the same instrument, run wider, not the same measurement.

harrier-270m F32, warm cache, profile-on, `english_short` corpus, one worker:

| preset | RSS lifetime MiB | PSS peak MiB | USS peak MiB | PSS file-backed peak MiB | PSS anon peak MiB |
|---|---:|---:|---:|---:|---:|
| `layer` | 105.52 | 103.34 | 101.98 | 27.50 | 75.85 |
| `attn-ffn` | 99.77 (-5.5%) | 97.64 (-5.5%) | 96.29 (-5.6%) | 21.80 (-20.7%) | 75.85 (+0.0%) |
| `budget:10MiB` | 95.72 (-9.3%) | 93.68 (-9.3%) | 92.32 (-9.5%) | 17.83 (-35.2%) | 75.85 (+0.0%) |
| `unit` | 96.48 (-8.6%) | 94.32 (-8.7%) | 92.96 (-8.9%) | 18.47 (-32.8%) | 75.85 (+0.0%) |

| preset | p50 ms | p90 ms | p99 ms | items/s |
|---|---:|---:|---:|---:|
| `layer` | 349.20 | 650.53 | 1156.46 | 2.388 |
| `attn-ffn` | 352.45 (+0.9%) | 651.88 (+0.2%) | 1158.22 (+0.2%) | 2.375 (-0.6%) |
| `budget:10MiB` | 361.07 (+3.4%) | 672.77 (+3.4%) | 1161.98 (+0.5%) | 2.316 (-3.0%) |
| `unit` | 379.21 (+8.6%) | 686.81 (+5.6%) | 1179.87 (+2.0%) | 2.246 (-6.0%) |

**The file-backed component moves in the direction the leased-byte table
predicts; total RSS does not, because it is not the dominant term.**
`pss_file` — the resident share of the mmap'd GGUF weights — falls 27.50 to
17.83 MiB (-35.2%) from `layer` to `budget:10MiB`, tracking the 53.0% leased-byte
reduction in the same direction if not the same magnitude (the gap is the
common-class ranges retained for the model's lifetime plus page rounding at
each new group boundary). `pss_anon` — the activation slot store and the
`ggml_gallocr` compute buffer — sits at 75.85 MiB across every preset,
identical to two decimal places, because none of it is what partitioning
touches. Anon is roughly triple the size of the weight footprint it is compared
against, so a large cut in leased weight bytes is a small cut in total RSS. The
`layer` preset here (105.52 MiB) also lands close to the M4 matrix's own
profile-on Harrier F32 streaming row (102.39 MiB lifetime, 100.38 MiB PSS peak,
98.96 MiB USS peak) despite the wider corpus, which is the cross-check that
this instrument reproduces the M4 measurement.

Latency moved the other way from residency: `budget:10MiB` and `unit` are both
slower than `layer` by throughput (-3.0% and -6.0%), and `unit`'s p50 is 8.6%
worse. `attn-ffn` is within noise of `layer` (+0.9% p50, -0.6% throughput) for a
real residency win, which is the shape a "cut once, keep it coarse" default
would take if one were chosen from this data. Profile-on latency remains
diagnostic — the sampler's own `smaps_rollup` read cost ranged 0.22-72.78 ms per
sample across these runs — so these p50/p90/p99 figures describe relative order
between presets, not the profile-off numbers a throughput claim would use.

## Cost side

Nothing here measures latency or throughput beyond the profile-on figures just
above. Three counters indicate the shape of the cost and are reported so the
trade is visible rather than implied:

- `willneed`/`dontneed` roughly quintuple from `layer` to `unit`. The bytes read
  are unchanged; the syscall and page-table-teardown count is not. These are
  totals over the test's workload — 8 serial embeds plus a two-context
  concurrency block — so they compare presets, but are not per-sentence rates.
- `gallocr replans` rises from 24 to 1160 on harrier F32. `ggml_gallocr` re-plans
  whenever a graph's node or leaf count differs from the previous one, and a
  finer partition makes consecutive graphs differ almost every time. Only the
  node count is reachable through ggml's public API, so this is a lower bound.
- `slot KiB` is host memory held for values crossing group boundaries. It rises
  from 77 to 899 KiB on harrier F32 and grows with sequence length, while the
  weight bytes saved are constant. The two move in opposite directions, which is
  why both are in the same table.

## Limits specific to this addendum

- Different container from the M4 matrix throughout; the RSS/PSS/USS section
  uses the same measurement class as M4 (`smaps_rollup`, 10 ms interval) but a
  different container and a wider corpus, so it is evidence of the same shape
  of effect, not a slot-in replacement for an M4 row.
- Leased bytes are a request to the kernel, not a measurement of residency; that
  is exactly why the RSS/PSS/USS section exists, and why the two disagree on
  magnitude (35% observed file-backed reduction vs. 53% leased-byte reduction
  at `budget:10MiB`, and total RSS down only 9.3%).
- The peak cannot fall below one GGUF tensor, because a unit's weight list names
  whole tensors. For harrier that floor is `ffn_gate` plus `ffn_up` — 10,240 KiB
  — since `ggml_geglu_split` fuses them and both must be live at the multiply.
  This is why `unit` and `budget:10MiB` reach the same F32 peak from different
  group counts, and the same file-backed PSS peak within noise.
- Accuracy figures used the 8 golden fixture sentences; the RSS/PSS/USS and
  latency figures used the 100-sentence `english_short` corpus group, 10 timed
  passes each. The two do not share a sample set.
- Sequence length in both was whatever `english_short` and the golden fixtures
  tokenize to — short. The slot-versus-weight trade reverses at long inputs
  (documented in `## Leased weight bytes by preset` above) and that crossover
  is not measured here for RSS/PSS/USS.
- One run, no repetitions, no confidence interval anywhere in this addendum.
  The leased-byte columns are deterministic properties of model and preset; the
  RSS/PSS/USS, latency, and counter columns are not.

## Saved evidence (addendum)

- `bench/results/partition-sweep/single_short_harrier_f32_streaming.json` — `layer`
  preset raw result: full `nanoembed-bench` schema-v2 JSON via `bench/runner.py`,
  `--memory-profile --memory-profile-interval-ms 10`, warm cache, 100-sentence
  `english_short` corpus, 10 iterations (1,000 timed requests, 500 warmup).
- `bench/results/partition-sweep/single_short_harrier_f32_streaming_attn_ffn.json`,
  `..._unit.json`, `..._budget10m.json` — the same run for the other three
  presets. All four share corpus, cache state, threads, and sampling interval;
  `partition` is the only field that differs, so the differences above are
  attributable to it (`bench/runner.py`'s `AB_CONTROLLED_FIELDS` covers
  `partition` for exactly this reason).
- Not covered by `SHA256SUMS`: these four files were added after the manifest
  was generated for the M4 bundle proper and are evidence for this addendum
  only, not for the M4 closeout above it.

---

# Addendum: tokenizer metadata release and BPE disk index (2026-08-27)

This is a post-M4 memory change, not a rerun of the M4 benchmark matrix. It
uses the Linux-only `nanoembed-tokenizer-memory-probe` construction instrument
against Harrier F32 in the current arm64 Docker build environment. Its memory
and timing values must therefore not be substituted into the preserved M4
eager/streaming tables above.

## Outcome

Harrier's SentencePiece-BPE merge table no longer remains as an in-process
`unordered_map`. The tokenizer keeps its existing `char_ix_` character map,
but writes the resolved merge rules to a content-addressed local index on the
first load and uses that index on later loads. Large tokenizer source arrays in
the GGUF context are removed immediately after both architecture and tokenizer
construction have completed.

The implementation changes no GGUF file, model distribution, or public C ABI.
The only external setting is `NANOEMBED_CACHE_DIR`; absent that override, the
cache follows the OS user-cache location. Cache eviction is intentionally not
automatic in v1: a user may delete the NanoEmbed cache directory while no
NanoEmbed process is using it.

## Index and lifetime design

- Identity is SHA-256 over separately framed `tokens` and `merges` arrays:
  array tag, count, and each string's length and bytes. The file is named
  `bpe-merges-v1-<sha256>.idx`, so F32, Q8_0, and Q4 variants with the same
  tokenizer share one cache.
- A fixed 4 KiB header holds the magic/version/little-endian marker, tokenizer
  digest, payload digest, record/page counts, and section offsets. Each data
  page is exactly 4 KiB: a 16-byte page header and at most 255 16-byte
  `{pair_key, rank, merged_id}` records sorted by pair key. Duplicate pairs
  retain the lowest merge rank, matching the former `unordered_map::emplace`
  semantics.
- Only the page fence table is permanent RAM. Harrier has 2,020 pages and a
  32,320-byte fence. Lookup binary-searches that fence, performs one explicit
  4 KiB offset read, then binary-searches the page. `mmap` is not used for the
  index; POSIX uses `pread`, Windows uses explicit-offset `ReadFile` with
  `OVERLAPPED`.
- Each encode owns an independent eight-page (32 KiB) cache. The file handle
  and fence are read-only, with no shared mutable page buffer or mutex, so
  distinct contexts continue to encode concurrently.
- On a warm cache hit, construction hashes source metadata and builds only
  `char_ix_`; it does not construct the temporary full token index or merge
  record array. On a miss, those temporary `string_view` structures live in a
  dedicated PMR arena backed by anonymous OS mappings (`VirtualAlloc` on
  Windows), then the entire arena is returned after the atomic cache write.
- Header/source/payload validation happens before GGUF tokenizer metadata is
  discarded. Every page also has a CRC checked during encode, so an in-place
  post-load corruption or short read raises `TokenizerError` instead of
  returning a plausible wrong token ID. Invalid cache files are regenerated
  through a unique temporary file plus atomic replacement. If generation or
  replacement cannot succeed, model loading fails; there is no high-memory
  hash-map fallback.

## Construction probe results

The previous `after_metadata_discard` Harrier measurement was
`Pss_Anon = 51,956 KiB`. The new probe observed the following:

| state | Pss_Anon KiB | merge cache hit | tokenizer load ms |
|---|---:|---:|---:|
| cold cache, after metadata discard | 7,064 | 0 | 615.264 |
| warm cache, after metadata discard | 7,064 | 1 | 240.386 |

The warm result is 44,892 KiB (about 43.8 MiB) below the prior measurement,
meeting the 20 MiB reduction acceptance floor and the <=32 MiB target. Cold
and warm steady-state `Pss_Anon` were identical, so the cache-build peak did
not remain resident after construction.

The Harrier index was 8,310,784 bytes (about 7.93 MiB), below the 9 MiB limit;
the 32,320-byte fence is below the 64 KiB permanent-index limit. For the probe
sentence, encode p50/p95 were 1,160.04/1,334.67 microseconds and averaged 94
page reads. These latency figures are recorded diagnostics, not a release gate.
Opening Harrier Q8_0 after Harrier F32 with the same cache directory produced a
cache hit, confirming cross-quantization tokenizer sharing.

## Full-process RSS/PSS/USS sample distributions

The construction probe above isolates tokenizer lifetime. This table uses the
existing `nanoembed-bench` profile-on instrument instead: Harrier F32,
`english_short` 10/100 selected samples, warm BPE cache, 10 ms
`smaps_rollup` cadence, and the same current Linux arm64 Docker build for all
execution modes and streaming partition presets. Each row is one independent
run with the same cache/corpus contract. Values are total-process MiB, not
`Pss_Anon` or a tokenizer component. Profile-on latency is diagnostic and is
not used here.

`peak sampled` is the maximum of the same samples from which the percentiles
are calculated. RSS additionally has a kernel `VmHWM` lifetime peak and a
post-warmup window peak; those are reported separately because PSS/USS have no
kernel high-water mark.

| execution | metric | peak sampled MiB | p50 MiB | p90 MiB | p95 MiB | p99 MiB |
|---|---|---:|---:|---:|---:|---:|
| eager | RSS | 1055.54 | 1055.54 | 1055.54 | 1055.54 | 1055.54 |
| eager | PSS | 1053.35 | 1053.35 | 1053.35 | 1053.35 | 1053.35 |
| eager | USS | 1051.83 | 1051.83 | 1051.83 | 1051.83 | 1051.83 |
| streaming (`layer`) | RSS | 36.46 | 26.39 | 35.29 | 36.35 | 36.43 |
| streaming (`layer`) | PSS | 34.27 | 24.20 | 33.09 | 34.16 | 34.24 |
| streaming (`layer`) | USS | 32.75 | 22.68 | 31.57 | 32.64 | 32.72 |
| streaming (`attn-ffn`) | RSS | 30.77 | 21.09 | 29.36 | 30.66 | 30.74 |
| streaming (`attn-ffn`) | PSS | 28.58 | 18.90 | 27.17 | 28.47 | 28.55 |
| streaming (`attn-ffn`) | USS | 27.06 | 17.38 | 25.65 | 26.95 | 27.03 |
| streaming (`budget:10MiB`) | RSS | 37.02 | 31.07 | 35.44 | 36.94 | 36.99 |
| streaming (`budget:10MiB`) | PSS | 34.83 | 28.87 | 33.25 | 34.75 | 34.80 |
| streaming (`budget:10MiB`) | USS | 33.31 | 27.36 | 31.73 | 33.23 | 33.28 |
| streaming (`unit`) | RSS | 27.45 | 20.43 | 25.88 | 27.37 | 27.41 |
| streaming (`unit`) | PSS | 25.25 | 18.23 | 23.69 | 25.17 | 25.22 |
| streaming (`unit`) | USS | 23.73 | 16.71 | 22.17 | 23.66 | 23.70 |

| execution | RSS lifetime peak MiB | RSS post-warmup window peak MiB | effective samples |
|---|---:|---:|---:|
| eager | 1078.54 | 1055.40 | 3,381 |
| streaming (`layer`) | 55.09 | 36.19 | 3,311 |
| streaming (`attn-ffn`) | 55.12 | 30.61 | 3,421 |
| streaming (`budget:10MiB`) | 55.03 | 36.97 | 3,738 |
| streaming (`unit`) | 55.13 | 27.35 | 3,843 |

The eager sample distribution is flat because the fully loaded F32 model
dominates the process footprint throughout the measurement window. All
streaming presets have a much smaller steady-state footprint, while their
p50-to-peak spreads reflect weight-page and execution residency during the
request sequence. In this single-run sample, `unit` has the lowest sampled
peak, `attn-ffn` is next, and `budget:10MiB` is slightly above `layer`; this is
a descriptive observation, not a preset selection or latency conclusion.
These are new post-index measurements, not replacements for the preserved M4
matrix: the container/build and corpus selection differ, and the cache-miss
construction peak remains covered by the separate construction probe above.

## Correctness and coverage

- SHA-256 standard vectors; header version/endian/source-digest damage;
  payload damage; partial final page; present/absent lookup; duplicate-rank
  selection; runtime corruption; truncation/short read; and unusable cache
  root are covered by tokenizer unit tests.
- Concurrent first creation is exercised with four threads on every platform
  and four processes on POSIX; no temporary partial file is accepted or left
  behind.
- The real tokenizer fixtures remained exact: BERT 100/100 and Harrier
  132/132 token-ID sequences matched. Linux also passed tokenizer,
  distinct-context concurrency, mode-selection, streaming integration, mapped
  weight-store, and Harrier mapped-preparation tests.
- macOS and Linux builds were run locally. CI now retains the existing real
  Harrier cold/warm tokenizer runs on Ubuntu and macOS, and adds a Windows
  synthetic tokenizer job covering build, disk-index behavior, corruption, and
  concurrency. Windows was not run locally for this addendum.

The narrow measurement tool is `nanoembed-tokenizer-memory-probe`; it reports
construction snapshots plus cache hit, cache/fence bytes, load time, encode
p50/p95, and page-read count. Run it twice with the same
`NANOEMBED_CACHE_DIR` to compare cold and warm construction.

The supporting profile-on artifacts are
`post-bpe-disk-index-harrier-f32-eager-memory.json` and
`post-bpe-disk-index-harrier-f32-streaming-memory.json` (`layer`),
`post-bpe-disk-index-harrier-f32-streaming-attn-ffn-memory.json`,
`post-bpe-disk-index-harrier-f32-streaming-budget10m-memory.json`, and
`post-bpe-disk-index-harrier-f32-streaming-unit-memory.json` in this result
directory. They are post-M4 evidence and are not covered by the original
`SHA256SUMS` manifest.
