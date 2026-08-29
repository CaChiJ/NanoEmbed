# M4 benchmark and streaming execution plan (temporary)

This is the temporary source of truth for the benchmark preparation and M4
implementation work. It is deliberately split into agent-sized, sequential
stages. Only one sub-agent may work at a time. The root agent reviews and
verifies a stage before starting the next one.

The document is temporary in the sense that it tracks an active implementation.
Once M4 is complete, durable design and usage information must be moved into
`PLAN.md`, `README.md`, and `docs/guide/26080401/07-tests-bench-workflow.md`,
then this file may be removed in a separate cleanup.

## 1. Locked decisions

1. The streaming CLI/config option is added with the M4 implementation, not as
   a non-functional placeholder during benchmark preparation.
2. Performance is measured under two cache regimes: cold start and warm start.
3. Detailed memory sampling is a `memory-profile` on/off switch, not a separate
   benchmark program or scenario type.
4. With memory profiling off, RSS is still reported using cheap boundary reads:
   current RSS before GO and `VmHWM` before/after the measured window. No
   `smaps_rollup` sampling runs during the timed section.
5. With memory profiling on, PSS/USS are sampled at a configurable, shorter
   interval. Latency from a profiled run is diagnostic, not the authoritative
   performance result.
6. Accuracy reports cosine similarity plus absolute, RMS, and norm errors.
7. PyTorch golden data carries reproducible provenance.
8. Multiple independent benchmark runs and confidence intervals are deferred.
   Single-run result content and fingerprints are expanded instead.
9. Corpora are grouped by workload type, and callers can select N samples per
   group so quick and final runs use the same harness.
10. Throughput is named `single_request_items_per_sec`.
11. Existing regression gates remain unchanged for now. Newly added metrics are
    report-only unless an existing correctness test already gates them.

## 2. Scope and non-goals

In scope:

- Benchmark contracts, corpus selection, cold/warm execution, optional detailed
  memory profiling, richer statistics and fingerprints.
- Reproducible PyTorch reference fixtures and expanded vector-error reporting.
- A Linux M3.6 in-memory baseline produced by the improved harness before M4
  changes model loading.
- M4 mmap-backed weights, layer residency control, `use_streaming=1`, and
  in-memory/streaming A/B measurements.

Out of scope unless a later explicit decision changes it:

- M5 true batch execution and concurrent throughput.
- New performance, PSS, or USS hard gates.
- Five-run confidence intervals and automated statistical significance tests.
- A public tokenizer-count API solely for benchmark reporting.
- macOS emulation of Linux `/proc` memory metrics.
- Unrelated refactors of model architecture or forward operators.

## 3. Sequential agent protocol

For every stage:

1. The root agent starts exactly one sub-agent with that stage's section and the
   current repository context.
2. The sub-agent inspects existing changes before editing and does not discard
   work from previous stages.
3. The sub-agent edits only the files justified by its stage. Tempting cleanup
   is reported, not bundled.
4. The sub-agent runs the stage-specific verification and records:
   - files changed;
   - commands run and results;
   - decisions or deviations;
   - remaining risks.
5. The root agent reviews the diff, runs proportional verification, and updates
   the status table below.
6. Only after the stage is accepted may the next sub-agent be started.

No two sub-agents may run concurrently for this plan.

## 4. Status and handoff log

| Stage | Status | Owner | Evidence / handoff |
|---|---|---|---|
| A1. Result contract and statistics | accepted | sub-agent A1 | Schema v2, requested/resolved settings, nullable metrics, latency/window-throughput statistics and v1 comparison compatibility. Root verified 12/12 CTest, Python compatibility tests, M3→M3.5 comparison, and `git diff --check`. Linux binary runtime remains for A7/A8. |
| A2. Corpus groups and bounded selection | accepted | sub-agent A2 | Six manifest groups, deterministic hash-ranked `NAME[:N]` selection, per-group metadata/hash, explicit duplicate policy and temporary-file cleanup. Root verified 13/13 CTest, 16 Python benchmark tests, `py_compile`, and `git diff --check`. `vocab_spread` remains intention-based until tokenizer-row instrumentation exists. |
| A3. Optional memory profiling and cheap RSS | accepted | sub-agent A3 | Off-by-default profile switch; profile-off statm/VmHWM-only boundaries; profile-on rollup baseline/avg/peak/final, breakdowns, sampler diagnostics, and on/off self-tests. Root verified 14/14 CTest, 18 Python tests, syntax/build checks, and profile-off control flow. Linux `/proc` runtime remains mandatory in A7/A8. |
| A4. Cold/warm execution | accepted | sub-agent A4 | `cold|warm`, per-input fresh cold native/worker, phase/resource scopes, Linux fadvise+mincore verification, strict failure and cold aggregation. Root verified 16/16 CTest, 22 Python tests, and lifecycle/aggregation semantics; agent also verified cache control under Linux Docker. End-to-end GGUF cold runs remain A7/A8. |
| A5. Fingerprints and result aggregation | accepted | sub-agent A5 | Full code/ggml/binary/model/manifest/scenario/build/environment identities, requested/resolved run settings, bounded descriptive group/overall summaries, and an optional hashed raw-latency sidecar. Fingerprint mismatches remain diagnostic and gates are unchanged. Root verified 17/17 CTest, 30 Python benchmark tests, v1 M3→M3.5 comparison, timing-path placement, and `git diff --check`. Linux-only optional fingerprint fields remain for A7/A8. |
| A6. Golden provenance and vector error | accepted | sub-agent A6 | Exact HF snapshot revision, pinned golden environment, deterministic CPU/FP32 generation contract, hashed provenance sidecars, expanded per-sample/aggregate error JSON, and F32-vs-quantization separation. Existing gates are unchanged; Q4 is report-only. Legacy fixtures are explicitly `legacy_unverified`. Root verified 18/18 CTest, 37 Python tests, accuracy JSON shape, valid integrity, fixture/manifest tamper rejection, legacy comparison, and `git diff --check`. Verified fixtures still require later explicit regeneration from exact revisions. |
| A7. Harness integration, tests, and docs | accepted | sub-agent A7 | Added portable cross-feature CLI integration tests and explicit non-Linux selftest skips; documented cache/profile/corpus/schema/fingerprint/accuracy interpretation. Root verified the 4 integration tests and macOS 2-test skip behavior. Agent verified Docker Linux arm64 21/21 CTest plus real BGE warm profile-off/on, cold, and strict-cold smoke; profile-off made zero rollup reads. Docker smoke artifacts were temporary and are not baselines. |
| A8. Linux M3.6 baseline capture | accepted | sub-agent A8 | Preserved an explicitly Docker-Desktop-arm64-scoped pre-M4 baseline: 16 result JSONs, 16 raw sidecars, 32 canonical workloads across all four cache/profile modes, accuracy JSON, environment note and checksums. Root reverified all checksums, sidecar hashes, profile contracts, and cold 16/16 verification. It is valid only for same-VM/mount pre/post comparison, not target hardware; source was dirty and optional CPU policy/NUMA/model data were unavailable. |
| B1. M4 design proof and failure policy | accepted | sub-agent B1 | Chose direct CPU reads from one `MAP_PRIVATE|PROT_READ` mapping; rejected active-layer copies. Traced the anonymous-copy boundary and fixed mode-lock, range classes, lease/advice ordering, cleanup, loud-failure, and rollback rules in `docs/m4-streaming-b1-decision.tmp.md`. Root reran the proof on BERT F16 and Harrier F32/Q8/Q4; all tensor alignment/bounds/block-layout/mmap-vs-pread checks passed. Mixed Q4 layouts and non-numeric file ordering are explicit constraints. |
| B2. mmap-backed model weight ownership | accepted | sub-agent B2 | Added internal `MappedWeightStore`/`MappedModelPreparation` with same-inode metadata parsing, validated read-only borrows, explicit identity/bounds/type/alignment checks, and RAII cleanup. Existing eager Embedder/public gate are unchanged. Root reran synthetic negative/cleanup tests and BERT F16/Harrier F32/Q8/Q4 preparation 4/4; architecture/tokenizer binding succeeded and all mapped leaves had null backend buffers. Linux B2 5/5 also passed. |
| B3. Layer residency control and streaming execution | accepted | sub-agent B3 + root review | Internal Linux phase runner, exact range classification, row-local token leases, protected shared residency advice, failure poisoning, and distinct-context concurrency accepted. Root re-ran portable range tests and all four real-model Linux cases; eager parity and existing golden gates passed with no leaked leases/advice failures. Public selection remains for B4. |
| B4. Public API, benchmark A/B, and correctness | accepted | sub-agent B4 + root review | Metadata-only model handles, atomic first-context eager/streaming lock, strict no-fallback public selection, native/runner requested-resolved evidence, controlled scenario pairs, and full-vector eager/streaming accuracy accepted. Root re-ran ABI/API/claim tests, Linux golden 8-way accuracy, concurrency/residency, and verified warm/cold runner artifacts. |
| B5. M4 measurements and closeout report | accepted | sub-agent B5 + root review | Same frozen Release binary produced 40 result JSONs, 40 raw sidecars and 72 workload rows in the preserved Docker environment. Root independently rechecked 92/92 M4 and 35/35 baseline checksums, reconstructed all 36 feature and 32 drift pairs from source JSON, verified strict cold/profile contracts and deltas, and accepted the scoped closeout/durable docs. |

Allowed status values: `pending`, `in_progress`, `blocked`, `accepted`.

## 5. Benchmark preparation stages

### A1. Result contract and statistics

Goal: establish a versioned, explicit result contract before behavior changes.

Required work:

- Add a benchmark result `schema_version`.
- Separate requested and resolved settings where the worker can resolve them.
- Record cache regime and whether detailed memory profiling was enabled.
- Represent unavailable metrics as JSON `null` plus a collection status, never
  as a misleading numeric zero.
- Add latency count, min, max, mean, p50, p90, p95, p99, standard deviation,
  and median absolute deviation.
- Rename aggregate throughput to `single_request_items_per_sec`.
- Add fixed-item-window throughput statistics when enough samples exist. Keep
  aggregate throughput canonical; window percentiles are descriptive.
- Preserve backward comparison of existing baseline files. Do not change gate
  thresholds in `bench/compare.py`.
- Add pure unit coverage for statistics and JSON serialization where practical.

Likely files:

- `tools/nanoembed-bench/main.cpp`
- `bench/runner.py`
- `bench/compare.py`
- CMake/test files if a new pure test target is justified

Acceptance:

- Old M3/M3.5 JSON can still be displayed or receives a clear schema warning.
- New JSON distinguishes zero, unavailable, and not collected.
- Percentile semantics are documented and tested.
- Existing CTest suite remains green.

### A2. Corpus groups and bounded selection

Goal: support representative workload groups without forcing every run to use
the full corpus.

Required work:

- Add a corpus-group manifest and reuse existing corpora where suitable.
- Initial groups:
  - `english_short`;
  - `multilingual_short`;
  - `unicode_edge` (emoji, combining forms, RTL and unusual scripts);
  - `vocab_spread` (inputs intended to touch a broad token-embedding region);
  - `medium`;
  - `long_context`.
- Support repeatable `--group NAME[:N]` and a global
  `--samples-per-group N` override in the Python runner.
- Use deterministic seeded selection based on stable text identity, not simply
  the first N lines.
- Record group size, selected size, seed, selected IDs, and selection SHA-256.
- Make duplicate handling across groups explicit.
- Keep quick runs small and final runs configurable without separate code paths.

Likely files:

- `bench/scenarios.yaml`
- a new manifest under `bench/`
- corpus files under `tests/corpus/` or `bench/corpus/`
- `bench/runner.py`
- Python unit tests for selection

Acceptance:

- Same manifest, seed, and N yield the same selected inputs.
- Different groups can request different N values.
- Unknown/empty groups fail before launching the native benchmark.
- The result preserves per-group identity.

### A3. Optional memory profiling and cheap RSS

Goal: make normal performance runs low-interference while preserving exact RSS
peaks and enabling detailed PSS/USS report runs.

Required work:

- Add an off-by-default `--memory-profile` option.
- Add `--memory-profile-interval-ms` for detailed sampling.
- With profiling off:
  - do not create a sampler thread;
  - do not read `smaps_rollup` during the timed window;
  - obtain baseline RSS from `statm`;
  - retain lifetime and window peak RSS from `VmHWM` boundary reads.
- With profiling on:
  - sample `smaps_rollup` at the requested interval;
  - report RSS/PSS/USS baseline, average, sampled peak, and final;
  - include final in aggregation;
  - add available anonymous/file/private/shared breakdown fields;
  - record requested interval, effective sample count, valid-sample ratio, and
    sampler read-duration statistics.
- Treat profiled latency as diagnostic in output metadata.
- Extend the synthetic self-test for both profiling states and a short-lived
  memory spike.

Likely files:

- `tools/nanoembed-bench/main.cpp`
- `tools/nanoembed-bench/metrics.{h,cpp}`
- `CMakeLists.txt`
- native unit/self-tests

Acceptance:

- Profile-off timed execution performs no `smaps_rollup` reads.
- RSS remains available in profile-off output.
- PSS/USS are `null/not_collected` when off.
- Profile-on captures the synthetic allocation within documented sampling
  limitations.

### A4. Cold/warm execution

Goal: stop mixing first-use storage behavior with warmed steady-state behavior.

Required work:

- Add `--cache-state cold|warm`.
- Warm execution keeps the current broad shape: load, create context, warm the
  selected inputs, reset the window peak, then time repetitions.
- Cold execution uses a fresh worker for each selected first request. It records
  model load, context creation, first embed, and startup-to-first-result times.
- Before a cold worker starts, request model-page eviction using a Linux-native
  mechanism such as `posix_fadvise(POSIX_FADV_DONTNEED)`.
- Verify and record file-page residency before and after eviction using a
  read-only mechanism such as `mincore`.
- Record `cold_cache_requested`, eviction success, resident percentages, and
  `cold_cache_verified`.
- Never silently label an unverifiable warm-cache run as cold. Provide a strict
  failure path for final cold measurements.
- Keep page faults and actual block-layer read bytes separated from cache hits.

Likely files:

- `tools/nanoembed-bench/main.cpp`
- `bench/runner.py`
- `bench/scenarios.yaml`
- Linux-only tests/helpers

Acceptance:

- Only the first inference from each fresh cold worker contributes to cold
  latency distribution.
- Warmup samples never appear in warm latency statistics.
- Cache-verification failure is explicit and machine-readable.
- Cold cost is bounded by the group-N controls from A2.

### A5. Fingerprints and result aggregation

Goal: make a result file sufficient to determine whether two measurements are
meaningfully comparable.

Required work:

- Record NanoEmbed git SHA and dirty status.
- Record ggml submodule SHA, benchmark binary SHA-256, compiler identity/version,
  build type, and relevant CMake options when discoverable.
- Expand Linux environment data with CPU governor/policy, NUMA information,
  total RAM, and filesystem identity where available.
- Record model path, size and SHA-256.
- Record corpus manifest and selected-input SHA-256.
- Record UTC start time, all requested settings, and resolved pooling.
- Explicitly record `independent_runs: 1` and `confidence_interval: null`.
- Aggregate per-group and overall results without destroying raw group identity.
- Add optional raw-sample sidecar output rather than making normal JSON
  unboundedly large.
- Expand environment mismatch diagnostics, but do not change existing gates.

Likely files:

- `tools/nanoembed-bench/metrics.{h,cpp}`
- `tools/nanoembed-bench/main.cpp`
- `bench/runner.py`
- `bench/compare.py`

Acceptance:

- Model, input selection, code, binary and machine identity are present.
- Missing optional Linux data is explicit rather than fatal.
- Comparison explains fingerprint mismatches before printing deltas.

### A6. Golden provenance and vector error

Goal: make correctness results reproducible and sensitive to more than vector
direction.

Required work:

- Make golden generation resolve and record the exact Hugging Face revision.
- Pin/reference a reproducible Python environment for fixture generation.
- Force and record CPU, FP32, deterministic settings, seeds, batch size,
  pooling, normalization and truncation settings.
- Add a provenance manifest beside each golden fixture, including package
  versions, hashes and the generation command.
- Verify fixture/manifest hashes when loading.
- Add per-sample and aggregate:
  - cosine similarity;
  - maximum absolute error;
  - mean absolute error;
  - RMSE;
  - output norm difference and relative norm difference;
  - worst and p50/p90/p95/p99 summaries where meaningful.
- Keep current cosine gates. New error metrics are report-only for now.
- Keep F32 implementation agreement separate from Q8/Q4 quantization loss.
- Produce machine-readable accuracy JSON that can be merged into the final
  benchmark report without running error calculations inside timed sections.

Likely files:

- `tools/dump_golden.py`
- `tests/integration/golden_test.cpp`
- `requirements-dev.txt` and/or a dedicated lock file
- golden provenance manifests under `tests/fixtures/golden/`
- a small accuracy aggregation script/tool if needed

Acceptance:

- A fixture identifies exactly which model revision and software produced it.
- Existing fixtures have an explicit migration or compatibility path.
- CTest still enforces existing cosine thresholds.
- JSON exposes all new report-only error metrics.

### A7. Harness integration, tests, and docs

Goal: validate the combined harness before it becomes the basis for M4 claims.

Required work:

- Exercise representative combinations:
  - cold/warm;
  - memory profile off/on;
  - one and multiple corpus groups;
  - default and overridden N;
  - old and new result schemas.
- Confirm profile-off observation does not include `smaps_rollup` in the timed
  window using an injectable counter or Linux integration check.
- Confirm cold workers are recreated per selected input.
- Confirm warmup samples are excluded.
- Update benchmark usage and metric interpretation in README and guide docs.
- Document that profile-on latency is diagnostic and PSS/USS peaks are sampled.
- Document cold-cache verification and the single-independent-run limitation.

Likely files:

- `CMakeLists.txt`
- test files under `tests/` and/or `bench/tests/`
- `README.md`
- `docs/guide/26080401/07-tests-bench-workflow.md`
- `PLAN.md` only where durable milestone statements need correction

Acceptance:

- Full CTest passes on supported local platforms.
- Python benchmark tests pass.
- Linux-only checks have a clear skip/failure policy off Linux.
- Documented commands match actual CLI behavior.

### A8. Linux M3.6 baseline capture

Goal: preserve an improved in-memory baseline before M4 changes loading and
residency behavior.

Required measurements, subject to group-N limits:

- BERT F16 short and long.
- Harrier F32 English short, multilingual short and vocab spread.
- Harrier Q8 English short, multilingual short and vocab spread.
- Priority order when compute is constrained:
  1. warm/profile off;
  2. cold/profile off;
  3. warm/profile on;
  4. cold/profile on.

Failure policy:

- This stage requires a Linux host with the target model files.
- Do not fabricate or substitute macOS memory numbers.
- If the available host cannot verify cold eviction, preserve warm results and
  mark cold capture blocked with evidence.

Acceptance:

- Results include the enhanced fingerprints and selected-input hashes.
- Model files match the hashes that will be used for M4 comparison.
- Baseline files are preserved under `bench/baseline/` with a short environment
  note.

## 6. M4 implementation stages

### B1. M4 design proof and failure policy

Goal: choose the smallest viable mmap/ggml integration before changing model
ownership broadly.

Required work:

- Trace how the vendored ggml version loads GGUF tensor data and which APIs can
  expose file-backed data without a full anonymous copy.
- Prototype or prove tensor alignment, offsets, quantized tensor compatibility,
  backend ownership, mapping lifetime, and concurrent-context safety.
- Compare at least:
  - direct file-backed tensor data;
  - copying one active layer into a reusable backend buffer.
- State how common tensors, token embeddings and one active layer are retained
  or advised away.
- Define failure handling for mmap/open/advice failures and unsupported
  platforms. `use_streaming=1` must fail loudly rather than fall back silently.
- Record the chosen design and rollback boundary in this document before B2.

Acceptance:

- The chosen path works for F16/F32 and target quantized types in principle.
- Ownership and cleanup order are explicit.
- No implementation begins with unresolved tensor-lifetime assumptions.

B1 decision record and proof evidence:

- `docs/m4-streaming-b1-decision.tmp.md`
- `tests/unit/mmap_gguf_proof_test.cpp`

The selected path is direct read-only file-backed tensor data on the current
CPU-only inference backend, with model-shared residency leases. Active-layer
backend copies are rejected for M4 because they duplicate RSS, add a mandatory
copy, complicate concurrency, and cannot solve the sparse token-table case.
`use_streaming=1` has a loud-failure-only policy and never silently selects the
in-memory runner.

### B2. mmap-backed model weight ownership

Goal: load model tensor data without the current full anonymous allocation while
preserving metadata validation and the in-memory path.

Required work:

- Introduce the minimal mapping/weight-store abstraction justified by B1.
- Keep model mappings read-only and shareable across contexts.
- Preserve architecture and tokenizer validation before unsafe tensor access.
- Retain `use_streaming=0` as the in-memory comparison/fallback path.
- Add bounds, alignment, overflow, truncated-file and cleanup tests.
- Do not add layer eviction policy yet beyond what is needed to prove mapped
  inference.

Acceptance:

- Both model families load through the new streaming preparation path.
- In-memory behavior remains unchanged.
- Failure leaves no partially owned mapping or backend resource.
- Existing non-performance tests pass.

### B3. Layer residency control and streaming execution

Goal: keep common weights and the active layer resident while releasing pages
that are no longer needed.

Required work:

- Classify mapped tensor ranges into common, per-layer, and other lifetimes.
- Coalesce page-aligned advice ranges safely; never advise outside the mapping.
- Apply an explicit prefetch/needed policy for the current layer if B1 shows it
  beneficial, and `MADV_DONTNEED` after use where appropriate.
- Handle the large token-embedding table by relying on accessed-row locality,
  not by eagerly retaining the entire table.
- Ensure graph execution has completed before advising away pages.
- Preserve distinct-context concurrency and define synchronization for shared
  model advice operations.
- Add trace/diagnostic counters suitable for tests without polluting the public
  ABI if possible.

Acceptance:

- Streaming inference succeeds for BERT and Harrier F32/Q8 target paths.
- Residency operations are page-safe and ordered after computation.
- Concurrency tests do not expose use-after-advice or mutable shared scratch.
- Existing output remains within current golden thresholds.

### B4. Public API, benchmark A/B, and correctness

Goal: expose M4 deliberately and prove that measurements actually select it.

Required work:

- Implement `use_streaming=1`; retain `0` for in-memory execution.
- Add benchmark `--streaming` and scenario `streaming: true/false`.
- Record requested and resolved execution mode.
- Fail if streaming is requested but not active/supported.
- Run same-binary in-memory vs streaming vector comparisons.
- Run streaming vs PyTorch golden comparisons for F32 and quantized variants.
- Keep error calculations outside timed performance windows.
- Update limits/API/ABI and concurrency tests as needed without changing the
  frozen function signatures.

Acceptance:

- A result cannot claim streaming without machine-verifiable resolved mode.
- Same-binary A/B uses identical model, corpus, pooling, threads and build.
- Existing correctness gates pass; new vector metrics are reported.

### B5. M4 measurements and closeout report

Goal: collect honest M4 results and move durable conclusions out of this
temporary plan.

Measurement axes:

- model/quantization;
- corpus group and selected N;
- cold/warm;
- streaming off/on;
- memory profile off/on.

Required work:

- Run the profile-off result as the authoritative latency/throughput number.
- Run profile-on for final RSS/PSS/USS analysis and label its latency diagnostic.
- Compare M4 in-memory with M3.6 to detect unrelated harness/code drift.
- Compare M4 streaming with M4 in-memory for the feature effect.
- Report cache verification, page faults and I/O alongside cold latency.
- Report all accuracy metrics and provenance.
- Preserve current gates; discuss proposed future thresholds separately.
- Update `PLAN.md`, README and the benchmark guide with durable results and
  remaining risks.

Acceptance:

- Every reported result is traceable to model, corpus, binary and environment
  hashes.
- Cold and warm claims are not mixed.
- Profiled and unprofiled latency are not mixed.
- M4 correctness and memory behavior are supported by saved artifacts.

Post-B5 raw-memory schema hardening (2026-08-24, implementation complete;
root review and exact cold sampled-mean correction complete):

- Extend native raw schema 1 without removing its latency fields: an optional
  `memory_profile` object preserves baseline, every periodic attempt and final
  RSS/PSS/USS plus smaps breakdown, monotonic timestamp, GO-relative elapsed
  time, read duration and DONE marker time.
- Keep one GO=0 origin per native invocation. Baseline is before GO and may be
  negative; cold worker origins remain independent in the runner sidecar.
- Add RSS/PSS/USS sampled p50/p75/p90/p95/p99 to the main result using the
  existing lower `floor(q * (n - 1))` rule. The population is valid periodic
  plus final samples; baseline stays excluded, consistently with mean/peak.
- For cold multi-worker profile-on results, compute sampled mean and percentiles
  over the exact merged raw population. Do not average worker means when sample
  counts differ; baseline/final boundary means and sampled peaks retain their
  existing cross-worker semantics.
- Keep profile-off on its old performance path: no sampler, no internal raw
  request unless latency sidecar output was explicitly requested, null sampled
  memory quantiles, and `disabled`/null raw memory samples.
- The runner accepts legacy native schema-1 latency sidecars with no memory
  object as `unavailable_legacy`; current profile-on cold aggregation requests
  temporary native raw data to merge worker sample populations exactly.
- This follow-up does not rewrite or reinterpret the already checksummed B5
  artifact bundle; it applies to subsequently built benchmark binaries.

## 7. Residual risks carried through the plan

- `posix_fadvise` is advisory and filesystems may decline effective eviction;
  residency verification is therefore part of the result, not an assumption.
- Frequent `smaps_rollup` reads can alter timing and potentially stretch page
  residency; profile-on latency remains non-authoritative.
- A single independent run cannot quantify thermal or system-load variance;
  fingerprints improve diagnosis but do not replace repetition.
- PSS/USS have no kernel high-water mark, so sampled peaks remain lower bounds.
- Mapped file pages may appear private-clean in a single-worker run. Sharing
  claims require a later multi-worker experiment, not inference from one process.
- M4 advice behavior on a shared mapping must be checked under concurrent
  contexts; memory savings must not come at the cost of data-race-like thrash.
