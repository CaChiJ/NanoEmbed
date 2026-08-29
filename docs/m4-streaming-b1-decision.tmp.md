# M4 B1 streaming design proof and failure policy (temporary)

Status: accepted design candidate for B2/B3 implementation. This record does
not claim that streaming inference exists yet.

## Scope lock

Requested outcome: choose the smallest viable mmap/ggml path with file and line
evidence, prove it against the four target GGUF variants, and remove unresolved
ownership, cleanup, concurrency, and failure-policy questions before B2.

Non-goals in B1:

- no production loader, public ABI, accuracy gate, or benchmark-gate change;
- no inference through mapped weights;
- no large weight copy, layer runner, or residency implementation;
- no promise that `madvise` will enforce a hard RSS limit (it is a kernel hint).

## Current load, compute, and ownership trace

1. `nanoembed_load_model` constructs the opaque model and its `Embedder`
   (`src/api/c_api.cpp:80-83`, `src/api/c_api.cpp:117-125`). The public model is
   therefore the lifetime root for architecture, tokenizer, and weights.
2. `Embedder` first calls `create_model_arch`, whose registry reads
   `general.architecture` through a metadata-only `gguf_init_from_file` call
   (`src/embedder.cpp:174-179`, `src/arch/registry.cpp:18-36`). BERT and Gemma 3
   scanners independently repeat metadata-only parsing and validate required
   metadata, shapes, and named tensors (`src/gguf_scanner.cpp:62-73`,
   `src/arch/gemma3_arch.cpp:33-45`).
3. Inference then opens the file again with `no_alloc=false`
   (`src/embedder.cpp:180-188`). This is the full anonymous-copy boundary.
   Vendored GGUF adds the whole padded data-section size to a ggml context arena
   (`third_party/ggml/src/gguf.cpp:741-769`), creates an I8 blob in that arena,
   and reads the entire data section into it (`third_party/ggml/src/gguf.cpp:771-809`).
   It then points every weight tensor into that blob at its GGUF-relative offset
   (`third_party/ggml/src/gguf.cpp:811-841`).
4. `ggml_init` allocates the arena with `ggml_aligned_malloc` when no external
   buffer is supplied, records that the context owns it, and `ggml_free` returns
   it (`third_party/ggml/src/ggml.c:1567-1606`,
   `third_party/ggml/src/ggml.c:1619-1629`). `ggml_new_tensor_impl` normally
   places data after its tensor struct when allocation is enabled, but a
   `no_alloc` tensor starts with `data == nullptr` and `buffer == nullptr`
   (`third_party/ggml/src/ggml.c:1722-1779`). Thus the current model's anonymous
   data blob lives exactly until `model_ctx` is freed
   (`src/embedder.cpp:92-98`, `src/embedder.cpp:132-135`).
5. The tokenizer is built from the GGUF metadata and architecture weight fields
   are bound to tensors in the owning inference context
   (`src/embedder.cpp:190-191`; BERT binding at `src/arch/bert_arch.cpp:42-71`;
   Gemma 3 binding at `src/arch/gemma3_arch.cpp:184-217`). Graph nodes retain
   those tensor pointers.
6. Each public context owns a distinct `ComputeScratch`. It unconditionally
   constructs a CPU backend and a context-local graph allocator
   (`src/embedder.cpp:138-169`). Inference builds the graph, allocates only
   activations through that context's allocator, then synchronously calls the
   CPU backend (`src/embedder.cpp:241-298`). CPU kernels consume leaf data
   through raw `tensor->data` pointers; for example, matmul reads its weight
   rows directly (`third_party/ggml/src/ggml-cpu/ggml-cpu.c:1182-1223`).

NanoEmbed inference is CPU-only today. The vendored ggml build may compile
Metal, BLAS, or other backends, but NanoEmbed neither selects nor schedules one:
the only product backend initialization is `ggml_backend_cpu_init` and the only
  backend-specific configuration is `ggml_backend_cpu_set_n_threads`
(`src/embedder.cpp:145-158`, `src/embedder.cpp:258`). B2 must test this actual
execution mode rather than infer capability from which vendored targets happen
to compile.

The current product graph uses model weights only as sources to `get_rows`,
`mul_mat`, `mul`, and `add`; there is no in-place, copy-to-weight, tensor-set, or
optimizer operation under `src/forward/` or `src/arch/`. This source audit is a
required invariant for a `PROT_READ` store. B3 tests must catch any later graph
builder that attempts to make a mapped weight a destination.

### Vendored GGUF/API facts

- This ggml revision has no public `gguf_get_data` API. The public surface
  exposes `gguf_get_data_offset`, tensor offsets, types, names, and sizes
  (`third_party/ggml/include/gguf.h:88-127`). The private `gguf_context::data`
  field exists only for the allocating loader (`third_party/ggml/src/gguf.cpp:217-228`).
  B2 must use `mapping_base + gguf_get_data_offset() +
  gguf_get_tensor_offset()`, not a nonexistent/older API or a vendored struct
  layout.
- `general.alignment` defaults to 32 and must be a non-zero power of two
  (`third_party/ggml/include/gguf.h:44-46`,
  `third_party/ggml/src/gguf.cpp:560-567`). The data-section start is padded to
  it (`third_party/ggml/src/gguf.cpp:702-710`). Tensor offsets are relative to
  that start and are required to be the exact sequential, alignment-padded
  layout (`third_party/ggml/src/gguf.cpp:712-733`).
- The parser rejects negative/overflowing shapes, rows not divisible by their
  type block size, and unrepresentable byte sizes
  (`third_party/ggml/src/gguf.cpp:606-683`). `ggml_nbytes` computes the actual
  strided byte extent and treats block-quantized dimension zero specially
  (`third_party/ggml/src/ggml.c:1267-1289`). B2 must still repeat checked
  `data_offset + tensor_offset + nbytes <= fstat_size`: metadata-only parsing
  does not read the data blob and therefore is not sufficient evidence that a
  non-truncated file covers every mapped tensor.
- The relevant layouts are native ggml layouts, not dequantized copies: F32
  and F16 use block size 1; Q8_0 uses its packed block; Q4_K uses the K-block
  layout (`third_party/ggml/src/ggml.c:650-663`,
  `third_party/ggml/src/ggml.c:716-723`,
  `third_party/ggml/src/ggml.c:763-769`). The proof records exact compiled
  block/type sizes. The deployed Q4 file is mixed F32/Q4_0/Q4_K/Q8_0, so a
  path that only admits the label `Q4_K` is not compatible with that file.
- The CPU backend can wrap a non-owned host pointer and advertises that
  capability (`third_party/ggml/src/ggml-cpu/ggml-cpu.cpp:390-417`), but its
  wrapper interface also exposes `set`, `memset`, and `clear` operations that
  write through the pointer (`third_party/ggml/src/ggml-backend.cpp:2230-2248`,
  `third_party/ggml/src/ggml-backend.cpp:2281-2293`). Weight leaves in the
  current allocating GGUF path already have no backend buffer. The selected B2
  path therefore leaves mapped weight `tensor->buffer == nullptr` and sets only
  the read-only leaf `tensor->data`; it must never call a tensor-set/clear API on
  mapped weights.

## B1 proof

`tests/unit/mmap_gguf_proof_test.cpp` is a standalone proof, linked only to
vendored ggml. For each real file it:

1. opens and `fstat`s the file, maps it `MAP_PRIVATE | PROT_READ`, and parses a
   metadata-only GGUF context through a duplicated descriptor for the same
   opened inode;
2. checked-adds every absolute tensor range and verifies it is within the file;
3. rechecks GGUF and CPU alignment, canonical padded offsets, ggml type,
   `ggml_nbytes`, type size, block size, and row/block divisibility;
4. temporarily points each metadata tensor at the mapped bytes while retaining
   `buffer == nullptr`;
5. compares bounded samples from the beginning, middle, and end of every tensor
   against `pread` from the same absolute file offsets;
6. clears the borrowed pointers, frees tensor metadata, unmaps, then closes.

The test is intentionally sparse: it faults only up to 192 bytes per tensor and
does not create a second full model copy.

Observed macOS arm64 proof results (the same target is also registered as four
conditional CTests):

| File | File/data offset | GGUF/CPU alignment | Tensors and compiled layouts | Token embedding | Per-layer payload | Result |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| BERT F16 | 67,308,128 / 760,928 | 32 / 32 | 74 F16 (block 1, 2 B), 123 F32 (1, 4 B) | 23,440,896 B | 3,558,912 B × 12 | pass |
| Harrier F32 | 1,088,170,176 / 15,777,472 | 32 / 32 | 236 F32 (1, 4 B) | 671,088,640 B | 22,294,528 B × 18 | pass |
| Harrier Q8_0 | 300,796,192 / 15,777,568 | 32 / 32 | 127 Q8_0 (32, 34 B), 109 F32 | 178,257,920 B | 5,931,008 B × 18 | pass |
| Harrier Q4_K | 250,661,152 / 15,777,568 | 32 / 32 | 36 Q4_K (256, 144 B), 90 Q4_0 (32, 18 B), 1 Q8_0, 109 F32 | 178,257,920 B | 3,145,728 B × 18 | pass |

Every tensor's mmap samples matched positioned reads on both the portable
macOS arm64 build and an Ubuntu 24.04/glibc arm64 Docker build. The BERT file stores
layers lexicographically (`0,1,10,11,2,...`), not in numeric execution order.
B2/B3 must build ranges from validated tensor references/names and must not
infer a layer's file range from the next numeric layer's offset.

## Candidate comparison and decision

| Property | Direct read-only file-backed tensor data | Copy active layer to reusable backend buffer |
| --- | --- | --- |
| Cold latency | Page faults/readahead only; no mandatory full-layer copy | Mandatory full-layer read plus memcpy before every layer |
| Warm latency/throughput | CPU reads the file-backed cache directly | Adds memory bandwidth and pointer/rebind work |
| Peak RSS | Mapped pages that are actually touched; no duplicate layer | File pages and an anonymous active-layer copy can overlap |
| Token embedding | Natural sparse-row faults; no 671 MB allocation | A full-table buffer defeats the goal; row staging is a separate design |
| Complexity | Metadata context + checked pointer binding + advice ranges | Per-context buffers or serialized sharing, tensor aliases/rebinding, and copy scheduling |
| Concurrent contexts | Read-only weights share naturally; advice needs leases | A shared buffer serializes; per-context buffers multiply RSS |
| Backend portability | CPU-only; other backends must be rejected | Better basis for a future device-transfer path |
| Rollback | Isolated internal weight-store/runner branch | Larger execution and ownership rewrite |

Decision: for the current CPU-only scope, use direct read-only file-backed
tensor data and control only OS residency. Do not create a reusable active-layer
weight copy in B2/B3. It loses on both memory and CPU latency and is especially
incompatible with the sparse 671 MB token table. Reconsider staging only when
NanoEmbed deliberately adds a backend that cannot consume host file mappings;
that must be a new explicit execution mode, not a fallback hidden under
`use_streaming=1`.

## Selected lifetime and execution policy

### Model mode and rollback boundary

The current eager anonymous allocation must move behind context-mode selection;
otherwise a streaming context has already hit the full-model HWM before it can
request streaming. The first successfully created context fixes one internal
model execution mode:

- `use_streaming=0`: initialize the existing `gguf_init_from_file(no_alloc=false)`
  store and retain the current monolithic in-memory runner unchanged;
- `use_streaming=1`: on Linux only, initialize the validated read-only mapped
  store and the streaming runner;
- later contexts on the same model must request the same mode; a mixed-mode
  request fails loudly. Separate model handles are required for an A/B run.

Mode initialization is guarded by the model mutex. If initialization fails
before a context exists, RAII removes the partial store and the mode remains
unselected; an explicit later `use_streaming=0` request is permitted. There is
never an automatic fallback. This first-mode lock is the smallest way to keep
one bound `ModelArch`, share weights, preserve distinct-context concurrency,
and avoid simultaneously retaining mapped and anonymous stores.

The rollback boundary is internal: delete the mapped store/streaming runner
branch and have mode initialization select the existing `no_alloc=false` store.
The public ABI and in-memory graph math do not change. One measurement-visible
change is unavoidable and must be reported: weight I/O moves from
`nanoembed_load_model` to the first `nanoembed_new_context`; end-to-end
startup-to-first-result remains the comparable startup metric.

### Range classes

- **Common small tensors:** BERT position/type embeddings and embedding norm;
  Gemma output norm; any other non-layer small tensor that the architecture
  manifest validates. Coalesce their page-aligned tensor ranges, request them
  before their phase, and retain them for the model lifetime. The proof totals
  are about 0.4 MB for BERT and 2.5 KB for Harrier, so retaining them is cheaper
  and safer than repeated advice.
- **Token embedding:** never `WILLNEED` or copy the whole table. Compute exact
  row byte ranges from validated token IDs, tensor strides, and type blocks,
  page-align and coalesce those rows, and advise only them. A model-shared token
  table lease records active embedding phases plus pending row ranges; the last
  active embedding phase may `DONTNEED` the pending union after its embedding
  graph has completed.
- **Per-layer tensors:** obtain tensors from the architecture manifest, group by
  parsed numeric `blk.N.` name, page-align each tensor independently, and
  coalesce only overlapping/adjacent ranges for that layer. Acquire a layer
  lease, optionally `WILLNEED`, compute that layer graph, then release. The last
  lease holder issues `DONTNEED`. Never advise a min/max span that includes an
  unclassified gap.

The current monolithic graph cannot safely discard a layer immediately: all
layers execute inside one synchronous backend call (`src/embedder.cpp:265-293`).
B3 must segment execution into embedding, one graph compute per layer, and
final norm/pooling phases, with context-owned activation ping-pong storage. The
ordering invariant is strict:

```text
acquire shared range lease
  -> optional WILLNEED
  -> build/allocate phase graph
  -> synchronous CPU graph compute returns
  -> release lease
  -> if count is zero, DONTNEED pending ranges
```

No advice occurs merely because graph construction has advanced past a node.
For any future asynchronous backend, completion/synchronization would be
required before release; B2/B3 instead reject non-CPU backends.

### Distinct-context concurrency

Contexts keep independent backends, graph allocators, metadata arenas, and
activation buffers as today. They may compute concurrently against the shared
read-only mapping. A model-shared residency coordinator contains only short
critical sections for region lease counts, pending ranges, and advice calls;
the compute itself is never under the model mutex.

An advice operation and its counter transition occur under the same mutex. This
prevents one context from advising a layer away while another has an active
lease. It cannot prevent ordinary cache pressure or a third process from
changing page-cache state, so benchmark fingerprints and cold/warm scenarios
remain necessary.

### Cleanup order

The mapped store owns the open FD, mapping, GGUF metadata context, ggml metadata
context, and residency coordinator. Required destruction order is:

1. all public contexts/phase leases and graph computes have ended (existing API
   ownership requirement);
2. destroy the bound architecture/runner so no tensor pointer remains usable;
3. clear mapped tensor `data` borrows and `ggml_free` the metadata context;
4. `gguf_free` the GGUF context;
5. `munmap` the file; then close the FD used by advice/identity checks.

The ggml metadata context must never own or free the mapping. Partial
construction uses the reverse of these same RAII owners. The in-memory store
keeps its current order: free `model_ctx`, then `gguf`.

## Loud failure policy

`use_streaming` accepts exactly 0 or 1. For 1, all of these are errors surfaced
through `nanoembed_new_context`/`nanoembed_last_error`, never triggers for the
in-memory runner:

- non-Linux platform or non-CPU execution backend;
- open/dup/fdopen/fstat/mmap failure, zero/too-large file, or inode identity
  changing during preparation;
- GGUF parse/architecture/tokenizer/tensor validation failure;
- non-power-of-two or insufficient absolute alignment;
- offset/padding/addition overflow, truncated range, or mapping-size mismatch;
- any required tensor using a type outside the initially proven set
  F32/F16/Q8_0/Q4_0/Q4_K;
- mandatory `madvise` failure. Failure before compute aborts that embed. Failure
  releasing pages after compute makes the call fail (the output is not valid to
  the caller) and poisons the streaming residency coordinator so later calls
  also fail loudly until the model is recreated;
- `mincore` failure when a diagnostic/verification mode requests it. Normal
  execution need not call `mincore` in the timed path;
- `posix_fadvise` failure whenever it is explicitly used. The selected runtime
  does not need file-wide fadvise: it can interfere with concurrent contexts,
  while mapping-range advice is sufficient. The benchmark's separate cold
  control retains its own strict/non-strict policy.

Retry only interrupted open/read-style system calls where POSIX permits it.
`MADV_DONTNEED` remains a hint even when it returns success; this is why a
machine-verifiable resolved streaming mode plus measured RSS/PSS/USS is required
instead of claiming a hard memory guarantee.

External mutation/truncation of a mapped model can still produce inconsistent
bytes or SIGBUS after validation. B2 should retain the FD/inode/size identity,
document model files as immutable for a model handle's lifetime, and test
ordinary truncation before binding. Preventing hostile concurrent file mutation
is outside M4's current trust boundary.

## B2/B3 handoff invariants

B2 may start only with these fixed rules:

1. map one immutable file read-only; parse metadata without a data allocation;
2. validate architecture/tokenizer/names/shapes/types/alignment/checked bounds
   before assigning any mapped `tensor->data` pointer;
3. CPU raw mapped leaves have no backend buffer and are never written;
4. `use_streaming=0` calls the existing allocating GGUF path; no fallback from
   streaming is automatic or silent;
5. first context fixes a model's mode; same-mode contexts share weights;
6. cleanup follows the explicit borrow-before-owner order above;
7. range classification comes from validated tensor names/refs, never numeric
   file-order assumptions.

B3 then adds, without weakening B2:

1. phase-separated synchronous execution with context-local activations;
2. common, sparse-row token embedding, and per-layer advice classes;
3. model-shared leases so DONTNEED follows graph completion and the last user;
4. diagnostic counters outside the public ABI, with every syscall failure
   represented as an explicit streaming failure.

Residual risks to validate in B2/B3 are mapped end-to-end numerical parity,
actual Linux advice/residency behavior, page-aligned sparse-row effectiveness,
and latency interference under two same-model streaming contexts. The B1 proof
does not substitute for those tests.

## B2 implementation handoff

B2 implements the ownership and validation half of the selected design in
`src/mapped_weight_store.{h,cpp}` without selecting it from the public
`Embedder` or changing the `use_streaming` gate. `MappedWeightStore` owns one
regular-file descriptor, its retained device/inode/size identity, a
`MAP_PRIVATE | PROT_READ` whole-file mapping, and metadata-only GGUF/ggml
contexts. `MappedModelPreparation` owns the store first, then constructs the
`ModelArch` and tokenizer against those same contexts. Only after architecture,
tokenizer, required names and shapes have all validated does it publish mapped
`tensor->data` borrows and bind the architecture. Every mapped leaf retains
`tensor->buffer == nullptr`.

Preparation validates in this order: opened FD/path identity and non-empty
regular-file size; metadata-only parse through a duplicate of the same FD;
identity again; GGUF/CPU alignment and canonical tensor offsets; unique names;
metadata tensor presence; the B1-proven F32/F16/Q8_0/Q4_0/Q4_K type set and
block divisibility; ggml/GGUF byte-size agreement; checked
`data_offset + tensor_offset + nbytes` file bounds; architecture and tokenizer;
identity immediately before and after pointer publication; architecture weight
binding. B2 conservatively rejects an unsupported type on any stored tensor,
not only after it is selected as required. This is compatible with all four
proved deployment files and avoids publishing a partially bindable context.

Cleanup clears every borrowed `data` pointer, frees the ggml metadata context,
frees the GGUF context, unmaps, and closes the FD, in that order. Because the
store is the first member of `MappedModelPreparation`, reverse member
destruction drops tokenizer and architecture tensor references before store
cleanup. Constructor failures use the same RAII path; tests cover corrupt and
truncated files, arithmetic overflow/out-of-range ranges, unsupported types,
changed size identity, failed architecture preparation, and repeated
construction without descriptor growth.

Observed preparation results on macOS arm64 and Ubuntu 24.04/glibc arm64:

| File | Architecture | Tensors | Mapped bytes | Result |
| --- | --- | ---: | ---: | --- |
| BERT F16 | bert | 197 | 67,308,128 | architecture/tokenizer/binding pass |
| Harrier F32 | gemma3 | 236 | 1,088,170,176 | architecture/tokenizer/binding pass |
| Harrier Q8_0 | gemma3 | 236 | 300,796,192 | architecture/tokenizer/binding pass |
| Harrier Q4_K mixed | gemma3 | 236 | 250,661,152 | architecture/tokenizer/binding pass |

The model file remains a trusted immutable input while a model handle exists.
FD/path identity and size are checked during preparation and can be rechecked
without touching mapped pages, but hostile concurrent byte writes or truncation
after validation can still yield inconsistent reads or SIGBUS. B2 deliberately
does not add a copy/snapshot defense for that out-of-scope trust boundary.

B3 may consume `MappedModelPreparation` to add phase execution, range classes,
leases and advice. B2 contains no inference through the mapped context, no
`madvise`/`DONTNEED`, and no public streaming activation; end-to-end numerical
parity and residency behavior therefore remain B3 acceptance work.

## B3 implementation handoff

B3 implements an internal Linux-only runner in
`src/streaming_execution.{h,cpp}`. It is intentionally absent from the public
C ABI: `use_streaming=1`, benchmark `--streaming`, and requested/resolved mode
reporting remain B4 work. The eager `Embedder` still owns the allocating GGUF
path and its external behavior is unchanged.

`ModelArch` now exposes the smallest phase seam needed by that runner:
embedding, one numbered layer, final architecture transform, phase input
requirements, and an exact validated weight-name plan. Both eager
`build_graph` implementations compose those same phase methods. BERT runs
embedding, 12 encoder blocks, identity final; Gemma 3 runs token embedding, 18
blocks, and output RMSNorm. Pooling and optional L2 normalization form the
runner's final graph.

Every phase has a fresh context-local metadata graph and uses the context's
own CPU backend and gallocr. Hidden state crosses a phase boundary only through
two bounded context-owned F32 host vectors: synchronous graph completion,
backend read into one vector, then backend write into the next phase input.
Diagnostics report both the two-vector capacity and cumulative copy bytes; the
copy is not hidden as weight I/O. No full-model graph or shared mutable
activation/scratch exists.

Range classification consumes `MappedTensorInfo` records published by the B2
store (validated name, absolute offset, byte extent, row stride/count/type).
The BERT plan classifies exactly token embedding + four common tensors + 16
tensors per layer. Gemma 3 classifies token embedding + output norm + 13
tensors per layer. A missing, duplicate, or leftover mapped tensor fails model
initialization. Each tensor is bounds-checked first, page-aligned independently,
and only overlapping/adjacent page ranges are coalesced; numeric layer order is
never treated as file adjacency and disjoint gaps are never covered by a
min/max span.

Common ranges receive one mandatory `MADV_WILLNEED` and remain protected for
the mapped model lifetime. The token table itself is classified but never
advised wholesale. Validated token IDs select exact packed row-stride byte
ranges; duplicates are removed before page alignment/coalescing. A numbered
layer lease contains only that layer's classified ranges.

The model-shared residency coordinator performs count/pending/advice updates
under short critical sections while graph construction and compute stay
outside the mutex. An RAII lease must be marked synchronously complete before
release. The last region user coalesces its pending ranges, subtracts common
and all other active lease ranges (including cross-region shared pages), then
issues mandatory `MADV_DONTNEED`. Premature release is loud and poisons the
coordinator. A failed precompute WILLNEED fails that call; a failed postcompute
DONTNEED poisons the coordinator, propagates failure, and the runner fills the
caller output with NaNs before throwing. Normal inference does not call
`mincore`; an explicit internal diagnostic method exists only for tests.

Portable range tests cover unaligned/page-shared/adjacent/disjoint/mapping-edge
and overflow cases, duplicate/invalid token IDs, packed quantized row strides,
failure injection, common-page protection, and same-region concurrent leases.
The Linux integration test compares internal streaming against both the eager
same-binary path and existing PyTorch fixtures for BERT F16 and Harrier
F32/Q8, with Q4 report-only, then repeats inference through two distinct
contexts concurrently. It asserts zero active leases/failures/premature
release after execution and graph-completion counts equal release counts.

B4 must select this internal model/context explicitly, preserve no-fallback
failure semantics, and surface machine-verifiable requested/resolved mode. It
must not collapse the ping-pong copies into weight-load accounting or claim a
hard RSS limit from successful `madvise`. The B3 test API and diagnostics are
private implementation surfaces and must not be copied into the frozen C ABI.
