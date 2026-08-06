# NanoEmbed — 단계별 구현 계획

> 이 파일은 plan mode 산출물입니다. 승인 후 프로젝트 루트의 `PLAN.md`(또는 `docs/roadmap.md`)로 이동시켜 사용합니다.

---

## Context

NanoEmbed는 100MB~10GB 크기의 텍스트 임베딩 모델을 **수십 MB 수준의 런타임 RAM**으로 엣지 환경에서 돌리기 위한 C++ 라이브러리입니다. 현재 저장소는 README 체크리스트만 있는 빈 상태입니다.

핵심 메모리 절약 전략은 세 단계:
1. **어텐션 레이어 스트리밍** — 트랜스포머 한 레이어 분량의 가중치만 mmap된 GGUF에서 transient context로 로드해 forward 후 즉시 해제. 100MB~10GB 모델의 대부분은 가중치이므로 가장 큰 절약 포인트.
2. **레이어 단위 배치 처리** — 스트리밍은 호출당 I/O 비용이 크므로 N개 입력을 같은 레이어 윈도우 안에서 한꺼번에 흘려보내 분할상환. 이때 dominant cost가 가중치에서 인터-레이어 활성값(`[B, seq_len, hidden]`)으로 옮겨감.
3. **활성값 압축 (TurboQuant 등)** — 배치가 커질 때 활성값 메모리를 int8/int4로 압축. 사용자 명시 후순위.

**확정된 1차 결정 사항** (사용자 응답)
- 1차 타겟 모델: `bge-small-en-v1.5` (33M, 384-dim, 12-layer BERT)
- v0 아키텍처 범위: BERT 한 종류만 (RoBERTa/E5/DistilBERT은 future work)
- 양자화 가중치 지원 시점: **M4(스트리밍)부터** Q8_0/Q4_K_M 도입. M3 baseline은 F32/F16만.

**디폴트 가정** (필요 시 사용자가 뒤집을 수 있음)
- 빌드: CMake ≥ 3.18, ggml은 `third_party/ggml`에 git submodule
- 1차 플랫폼: macOS(arm64) + Linux(x86_64). Windows·모바일은 후순위
- 토크나이저: 인-트리 BERT WordPiece 구현 (~500 LOC)
- 풀링 디폴트: mean pooling + L2 normalize (CLS는 토글)
- 스레드: ggml thread pool은 M3부터 그대로 사용 (그래프 내부 병렬화). 호출 측 병렬화는 M8(바인딩)에서 다룸
- 파일 I/O: `mmap` + 레이어별 transient `ggml_context`. 사용 끝난 페이지는 `madvise(DONTNEED)`

---

## ggml 재사용 vs 우리가 짤 것

**ggml에서 그대로 가져다 씀**
- GGUF 파싱 (`gguf_init_from_file`, 메타데이터 KV, tensor info)
- 텐서 타입 시스템 (F32/F16/Q8_0/Q4_K_M ...)
- `ggml_context` / `ggml_cgraph` / `ggml_backend`
- 커널: matmul, softmax, layernorm, GeLU
- CPU 백엔드, macOS Accelerate / Metal

**직접 구현**
1. GGUF 텐서 이름을 BERT 슬롯에 매핑하는 스캐너
2. 레이어 1개분 가중치만 transient ctx로 복사하는 layer loader
3. BERT WordPiece 토크나이저 (vocab은 GGUF 메타데이터에서)
4. 인터-레이어 활성값을 보관하는 activation store (배치 모드의 메모리 핵심)
5. 레이어 0..N-1을 순회하는 streaming runner / batched runner
6. 풀링 + L2 정규화
7. 공개 C API (M1 동결) + Node 바인딩 (M8)
8. (M6) 활성값 int8 quant/dequant

---

## 소프트웨어 아키텍처

### 1. 레이어 구조 (의존성은 위→아래만)

```
┌─────────────────────────────────────────────────────────┐
│  Bindings        Node N-API (M8)                         │
├─────────────────────────────────────────────────────────┤
│  Public API      nanoembed.h (C ABI, M1 동결) / .hpp (M7) │
├─────────────────────────────────────────────────────────┤
│  Façade          Embedder — 토크나이저+Runner 조합        │
├─────────────────────────────────────────────────────────┤
│  Runtime         StreamingRunner · BatchedRunner          │
│                  (메모리 전략이 사는 곳)                    │
├─────────────────────────────────────────────────────────┤
│  Forward         GraphBuilders (embed/attn/ffn/pool)     │
│                  — 순수 ggml 그래프, 상태 없음              │
├─────────────────────────────────────────────────────────┤
│  Model           ModelManifest · LayerLoader              │
│                  · ActivationStore · Tokenizer            │
├─────────────────────────────────────────────────────────┤
│  Foundation      ggml · GGUF I/O · mmap · 텐서 유틸        │
└─────────────────────────────────────────────────────────┘
```

상위 레이어는 하위만 호출. 같은 레이어 안에서는 서로 헤더로 알지 않게 함 (예: GraphBuilders끼리 알지 않음). 이게 마일스톤별 슬라이싱과 단위 테스트를 단순하게 유지함.

### 2. 핵심 추상화

각 추상화는 **현 v0에 필요한 단 하나의 구현**만 가짐. 인터페이스를 미리 빼지는 않음 — 단, 헤더/책임을 분리해서 *나중에 인터페이스로 뺄 수 있게* 둠 (premature abstraction 회피).

- **`ModelManifest`** (M2) — GGUF 텐서를 BERT 인코더 슬롯에 매핑한 결과. 한 번 만들면 immutable한 값 객체.
- **`Tokenizer`** (M3) — `tokenize(string) -> {ids, attn_mask}`. v0는 WordPiece 단일 구현. 추상 인터페이스 X.
- **`LayerLoader`** (M4) — `load(transient_ctx, layer_idx) -> LayerWeights`. mmap된 영역의 가중치를 transient `ggml_context`에서 사용 가능한 형태로 만듦 (구현은 ggml view 또는 복사 중 택일, M4에서 결정). **stateless** (mmap 핸들과 manifest를 borrow).
- **`ActivationStore`** (M5) — `[B, seq_len, hidden]` 활성값 + attention mask 보유. v0는 F32 모드만, M6에서 int8 모드 추가 (같은 헤더에서 모드 분기).
- **`GraphBuilder` 함수들** (M3) — 각 함수가 `(ctx, inputs, weights) -> ggml_tensor*` 형태. 자유 함수로 시작. 입력의 leading 차원이 1이면 단일, B면 배치 — 함수는 동일.
- **`Runner`** (M4·M5) — `embed(input(s)) -> vector(s)`. `StreamingRunner`와 `BatchedRunner`를 별도 클래스로 분리하되 **같은 헬퍼**(LayerLoader, GraphBuilder, ActivationStore)를 공유. 차이는 외부 루프와 GraphBuilder에 넘기는 텐서의 batch 차원뿐 — 인코더 블록 그래프 자체의 토폴로지는 같음.
- **`Embedder`** (M3) — façade. 모델 크기·옵션을 보고 어떤 Runner를 쓸지 선택. 단일 thread-unsafe 핸들.

### 3. 데이터 플로우 (배치 모드)

```
vector<string>
   │
   ▼  Tokenizer
[B] {ids, attn_mask, length}                  ← 길이 버킷팅 (M5)
   │
   ▼  GraphBuilder.embed_layer  (persistent ctx)
   acts[B, S, H] = tok+pos+type embed → LN
   │
   ▼  Runner (Streaming or Batched)
for layer in 0..L-1:
   transient_ctx.reset()
   weights   = LayerLoader.load(transient_ctx, layer)
   acts_next = GraphBuilder.encoder_block(transient_ctx, acts, weights, mask)
   acts      = ActivationStore.swap(acts_next)  ← M6: 여기서 int8 압축/해제
   madvise(layer_region, DONTNEED)
   │
   ▼  GraphBuilder.pool + L2 norm
[B] vector<float>
```

### 4. 메모리 / 소유권 모델

라이프타임이 다른 4개 메모리 영역이 공존:

| 영역 | 라이프타임 | 크기 | 누가 소유 |
|---|---|---|---|
| **Persistent ctx** | 핸들 생존 동안 | 토큰 임베딩 테이블이 dominant (vocab × hidden × dtype). bge-small 기준 임베딩이 Q8_0이면 ~12MB, F16이면 ~23MB | `Embedder` |
| **Transient ctx** | 1 레이어 1 forward | 1 레이어 가중치 + 1 그래프 | `Runner`가 풀로 재사용 |
| **Activation store** | 1 `embed_batch` 호출 | `B × S × H × dtype` | `Runner` |
| **mmap region** | 핸들 생존 동안 | GGUF 파일 전체 (가상 주소). 실제 RSS는 OS가 페이지 단위로 관리 | `Embedder` |

**불변식 (M4 이후)**: 어느 시점이든 RSS = 임베딩 테이블 + 활성값 + 단 1 레이어 가중치 + 작은 상수. 이걸 `nanoembed-bench`의 RSS 측정 + Runner 단위 테스트로 어서션.

BERT는 모든 레이어 크기가 동일하므로 transient ctx는 한 레이어 분량으로 한 번만 잡고 매 레이어마다 `reset()` — `alloc/free` 반복 없이 단편화 회피.

### 5. 백엔드 / 동시성

- **백엔드**: ggml `ggml_backend_t` 그대로 사용. CPU 디폴트(macOS의 BLAS는 Accelerate 자동 링크). Metal/CUDA 백엔드는 빌드 옵션, NanoEmbed가 직접 이식하지 않음.
- **스레드**: 핸들은 thread-unsafe. 여러 핸들은 독립. ggml 내부 컴퓨트 스레드는 M3부터 사용. 호출 측 병렬화가 필요하면 핸들을 복제 — mmap이 공유되므로 추가 디스크 비용 없음.
- **Async**: C++ API는 동기. Node 바인딩만 worker thread로 감싸 Promise 노출 (M8).

### 6. 에러 모델

- **내부**: C++ 예외. 도메인 에러 (`InvalidGGUFError`, `MissingTensorError`, `TokenizerError`).
- **C ABI 경계 (M7)**: 예외를 catch해서 `nanoembed_status_t` 코드 + thread-local last error string으로 변환. 예외가 ABI를 넘는 일 없음.
- **Fail-fast**: 모델 로드 시점에 만나는 에러 (텐서 누락, 차원 불일치, 토큰 vocab mismatch)는 즉시 throw. Forward 시점에 발견되지 않게 함 — 그래야 운영 중 디버깅이 단순.

### 7. 확장 포인트 (지금 짓지 않되 *닫지도 않음*)

| 미래 요구 | 닫혀 있지 않은 자리 | 미래에 어떻게 |
|---|---|---|
| RoBERTa / DistilBERT / E5 | `ModelManifest`의 `arch` 태그 | manifest에 분기 + 신규 GraphBuilder 세트 |
| SentencePiece 토크나이저 | `src/tokenizer/` 디렉터리 분리 | Tokenizer 추상 인터페이스로 승격 |
| Metal / CUDA 백엔드 | ggml_backend 통일 | 빌드 플래그 + 런타임 선택 |
| 활성값 압축 (M6) | `ActivationStore`의 모드 분기 | int8/int4 모드 활성화 |
| 토큰 단위 임베딩 노출 | Runner는 이미 `[B,S,H]`까지 보유 | 공개 API에 `embed_tokens` 추가 |

원칙: **추상화는 두 번째 구현이 들어올 때 빼낸다.** v0는 자유 함수 + 구체 클래스로 시작.

### 8. 빌드 / 패키지 구조

```
            ggml (third_party submodule)
                     ▲
                     │ depends on
            nanoembed_core (static lib, src/ 전체)
              ▲          ▲           ▲          ▲
              │          │           │          │
        inspect      cli        bench       node binding
        (M1·M2)     (M3)       (M4+)       (M8)
```

- 코어는 **static lib 1개**. 도구·바인딩은 모두 거기에 링크. shared lib는 M7에서 옵션.
- 헤더 가시성: `include/nanoembed/` 만 PUBLIC, `src/` 는 PRIVATE — CMake `target_include_directories`로 강제.

### 9. 테스트 / 벤치마크 진입점 요약

세부 설계는 아래 「공개 API」, 「벤치마크」, 「정확도 테스트」 섹션 참조.

- **단위** — 토크나이저, GGUF 스캐너, LayerLoader (텐서 shape·dtype).
- **레이어 격리 정합성** — HF에서 저장한 *N번째 인코더 블록 입출력* 픽스처와 GraphBuilder 출력 비교 (수치 디버깅 1차 도구).
- **골든 정확도 (end-to-end)** — sentence-transformers + llama.cpp 두 oracle. 「정확도 테스트」 참조.
- **벤치마크 (메모리·CPU·I/O·latency)** — 마일스톤별 baseline JSON 비교. 「벤치마크」 참조.
- **ABI** — 순수 C 컨슈머 빌드 + 출력 일치 (M7).

---

## 공개 API — M1에서 동결, 이후 추가만

llama.cpp의 model/context 분리를 따름:
- **Model**: immutable, mmap된 GGUF, 여러 컨텍스트가 공유 가능.
- **Context**: 인스턴스 상태(설정·버퍼). thread-unsafe (병렬화는 핸들 복제로).

**시그니처 (`include/nanoembed/nanoembed.h`)**
```c
typedef struct nanoembed_model   nanoembed_model;
typedef struct nanoembed_context nanoembed_context;

typedef enum { NANOEMBED_POOL_MEAN, NANOEMBED_POOL_CLS } nanoembed_pool_type;

typedef struct {
    int n_threads;        /* 0 = auto */
    int max_batch;        /* 초과 시 자동 sub-batch */
    int max_seq_len;      /* 초과 시 뒤쪽 토큰 truncate */
    int use_streaming;    /* M3는 무시, M4+에서 적용 */
    nanoembed_pool_type pooling;
    int normalize;        /* L2 normalize */
} nanoembed_context_params;

nanoembed_context_params nanoembed_context_default_params(void);

nanoembed_model*   nanoembed_load_model(const char* path);
void               nanoembed_free_model(nanoembed_model*);
int                nanoembed_n_embed(const nanoembed_model*);
int                nanoembed_n_layer(const nanoembed_model*);

nanoembed_context* nanoembed_new_context(nanoembed_model*, nanoembed_context_params);
void               nanoembed_free_context(nanoembed_context*);

/* out 버퍼는 호출자 소유. 단일: n_embed. 배치: n_texts * n_embed. */
int nanoembed_embed(nanoembed_context*, const char* text, float* out);
int nanoembed_embed_batch(nanoembed_context*, const char** texts, int n, float* out);

const char* nanoembed_last_error(void); /* thread-local */
```
음수 리턴 = `NANOEMBED_ERR_*` 코드, `0` = OK.

**마일스톤별 채워지는 정도**
| | M1 | M3 | M4 | M5 | M7 |
|---|---|---|---|---|---|
| 시그니처 컴파일·링크 | ✅ | | | | |
| 단일 입력 임베딩 | stub | ✅ (전부 로드) | ✅ (스트리밍 가능) | | |
| 배치 (내부적으로 N회 단일) | stub | ✅ | ✅ | | |
| 배치 (실제 레이어 단위 배치) | stub | | | ✅ | |
| C++ RAII wrapper `.hpp` | | | | | ✅ |
| install / find_package / shared lib | | | | | ✅ |

**llama.cpp와 의도적으로 다르게 한 부분**
- `llama_batch` 자료구조 없음 — 시퀀스 간 의존 없는 임베딩에 과한 추상.
- KV 캐시 / `n_ubatch` 미노출 — encoder-only이므로 사용자에게 의미 없고, sub-batching은 내부 자동.
- 토큰 단위 출력 미노출 — 풀링 후만. 필요해지면 추가.

**확장 정책 (호환성 유지)**
- 새 옵션은 (a) `nanoembed_context_params` 끝에 필드 추가 + `default_params`가 디폴트 채움, 또는 (b) 새 함수 (`nanoembed_embed_v2` 류). 기존 시그니처 깨지 않음.
- 새 enum 값은 끝에만 추가. switch에 default 처리.

---

## 벤치마크

**원칙** 매 마일스톤 종료 시 동일 슈트 실행 → JSON을 직전 baseline과 비교 → 회귀 검출. 수치는 환경 의존이므로 **같은 머신에서만** 비교하며, 이를 사람 기억이 아니라 도구가 강제한다 (아래 「환경 고정」).

**플랫폼: Linux 전용.** 측정이 `/proc/<pid>/{status,statm,smaps_rollup,clear_refs}`에 전면 의존하므로 `nanoembed-bench`는 Linux에서만 빌드된다 (CMake에서 타겟 자체를 게이트). 라이브러리와 `ctest`는 계속 모든 플랫폼에서 빌드된다.

**측정 구조: fork + exec 격리.** 오케스트레이터(부모)는 모델을 로드하지 않는다. 자기 자신을 `--worker`로 `fork+exec`한 뒤 바깥에서 `/proc`으로 워커를 관찰한다.

- **`exec`까지 하는 이유**: 단순 `fork`는 부모가 더럽힌 페이지를 COW로 물려주고, 그 페이지가 자식 RSS에 그대로 잡힌다. `exec`은 주소 공간을 갈아엎어 워커를 ~4MB 바닥에서 출발시킨다. 실측(부모가 8MB 점유):

  | | 시작 RSS | 60MB 작업 후 |
  |---|---|---|
  | fork only | 8552 kB | 70192 kB |
  | fork + exec | **1152 kB** | **62796 kB** |

  RSS가 USS 못지않게 신뢰 가능해지는 게 요점이다 — M4 메모리 예산이 RSS로 서술돼 있기 때문.
- **측정 구간 정합**: `VmHWM`은 스스로 내려가지 않으므로 끝에서 읽으면 로드·warmup까지 포함한 peak이 나와, 측정 루프에 한정된 CPU/page-fault 델타와 **다른 구간**을 재게 된다. 창을 열기 직전 `/proc/<pid>/clear_refs`에 `5`를 써서 `VmHWM`을 현재 RSS로 리셋하고, 리셋 **전에** 한 번 읽어 두 구간을 모두 보고한다.

**지표**

| 지표 | 출처 | 구간 |
|---|---|---|
| `rss_peak_lifetime_mb` | `VmHWM` (리셋 전 판독과 최종 판독의 max). 커널 추적이라 정확 — 샘플 사이로 스파이크가 새지 않음 | 워커 전 생애 (로드 스파이크 포함) ★ **회귀 게이트** |
| `rss_peak_window_mb` | `VmHWM` (리셋 후) | 측정 루프 |
| `rss_baseline_mb` | 창 시작 시점 `smaps_rollup` | peak 해석용 바닥값 |
| `rss_avg_mb` / `rss_max_sampled_mb` | `/proc/<pid>/statm`, 50ms 샘플링 (~0.004ms/회) | 측정 루프 |
| `pss_*` / `uss_*` | `/proc/<pid>/smaps_rollup`, **500ms** 샘플링 | 측정 루프 |
| CPU user + sys | 워커가 `getrusage(RUSAGE_SELF)`로 자기 보고 (창 경계 델타) | 측정 루프 |
| Major / minor page faults | 워커 자기 보고 `ru_majflt` / `ru_minflt` | 측정 루프 |
| 디스크 read bytes | 워커 자기 보고 `/proc/self/io` `read_bytes` | 측정 루프 |
| Latency p50/p90/p99 (ms) | 워커가 `steady_clock`으로 호출별 기록 → 창 종료 후 파이프로 벌크 전송, 부모가 퍼센타일 계산 | 측정 루프 |
| Throughput (items/sec) | total items / wall time | 측정 루프 |

**RSS / PSS / USS를 모두 재는 이유** — 셋의 질문이 다르다: RSS는 "디바이스에 얼마나 비어 있어야 하나"(→ 게이트), PSS는 "워커를 여러 개 띄우면 총합이 얼마인가", USS는 "이 프로세스를 죽이면 얼마가 회수되나". M3에서는 모델이 익명 메모리로 올라가 셋이 거의 붙어 있고, **M4에서 GGUF를 mmap하면 갈라진다** — 그때를 위한 계측이다.

**두 종류의 peak** — `VmHWM`은 RSS에만 있다. PSS/USS는 커널 high-water mark가 없어 peak을 샘플 최대값으로 구하므로, 키 이름에 `_sampled`를 박아 이를 드러낸다. 또한 `smaps_rollup`은 대상의 `mmap_lock`을 잡고 페이지 테이블 전체를 걷는다 — 300MB 상주 기준 **약 6ms**(`statm`의 1400배)라, RSS와 같은 주기로 돌리면 측정 대상을 교란한다. 그래서 주기를 분리했다.

**환경 고정** — 모든 실행이 `environment` 블록(`kernel`, `cpu_model`, `nproc`, `page_size_bytes`)을 기록한다. `compare.py`는 baseline과 fingerprint가 다르면 경고하고 `--strict`에서는 비교를 거부한다(exit 2). 하드웨어 차이가 코드 변경으로 오독되는 걸 막기 위함이다.

**시나리오** (`bench/scenarios.yaml`) — 마일스톤이 지원하는 시점부터 슈트에 추가

- `single_short_f16` — M3+ — bge-small F16, batch=1, 짧은 문장 100개 코퍼스
- `single_long_f16` — M3.5+ — 같은 모델, 매 줄이 512토큰 캡을 채우는 장문 (M3의 고정 아레나로는 실행 불가였음)
- `single_short_q8_streaming` — M4+ — bge-small Q8_0, batch=1, streaming on
- `batch32_mixed_q8` — M5+ — 길이 분산 있는 500개, batch=32
- `batch128_mixed_q8` — M5+ — 같은 코퍼스, batch=128

**도구 체인**
- `tools/nanoembed-bench`: 시나리오 1개 실행 → JSON (stdout 또는 `--out`). `--selftest`는 GGUF 없이 하네스 자체를 검증
- `bench/runner.py`: `scenarios.yaml`의 모든 시나리오 실행 → `bench/results/<git_sha>.json` 으로 합침
- `bench/compare.py base.json cur.json`: 환경 fingerprint 확인 + 표 출력 + 회귀 시 non-zero exit

**Baseline / 회귀 정책**
- `bench/baseline/M3.json`, `bench/baseline/M4.json`, ... 마일스톤 통과 시점에 commit. 다음 마일스톤은 직전 baseline 대비.
- 임계 (`compare.py` 디폴트): latency ±15%, `rss_peak_lifetime_mb` 단조 비증가 (의도적 증가는 baseline 갱신 PR로 명시), major page faults는 마일스톤 의도와 일치 (M3→M4 증가 OK, M4→M5 *per-item* 감소 필수).
- PSS/USS는 **표에만** 싣고 게이트하지 않는다. 샘플링 기반이라 5% 임계를 걸기엔 노이즈가 크다. M4에서 mmap이 들어와 RSS 해석이 갈리면 재검토.
- 낡은 baseline은 지우지 않고 `bench/baseline/archive/`로 옮긴다 (`archive/README.md`에 왜 비교 불가인지 기록).

**CI 정책**
- 매 PR: `ctest`의 `bench_selftest` — 모델 없이 도는 유일한 벤치 측 검증. fork+exec 격리(워커 baseline이 부모 오염과 무관한가)와 `clear_refs` 리셋, 창 한정 RSS 증가분을 어서션.
- 성능 수치 자체는 **당분간** CI에서 재지 않는다. GGUF 다운로드·캐시 스텝이 아직 없고, 무엇보다 GitHub 공용 러너는 실행마다 배정 하드웨어가 달라(`ubuntu-latest`가 Intel/AMD를 오간다) `environment` fingerprint가 거의 항상 불일치한다 → 모든 PR이 비교 거부로 실패한다. 그 동안 **성능 baseline은 고정된 개발 머신에서 수동 측정**한다.
- (예정) **self-hosted runner**를 구성해 환경을 고정한 뒤 원래 계획대로 복귀: 매 PR 빠른 모드 + PR 코멘트, nightly full suite + main 브랜치 RSS 그래프. fingerprint가 고정되므로 커밋된 baseline과 직접 비교가 성립한다.
- 참고: 러너를 확보하더라도 게이트는 메모리 지표에 두는 편이 낫다. 같은 머신 반복 측정에서 `rss_peak`은 편차 0.15%로 결정적인 반면 latency는 부하에 따라 크게 흔들린다.

---

## 정확도 테스트

**Oracle 두 개를 함께 사용**

| Oracle | 검증하는 것 | 예상 일치 |
|---|---|---|
| **sentence-transformers** (HF, PyTorch fp32) | *의미적* 정확도 — "올바른 임베딩을 만드는가" | cosine ≥ 0.9999 |
| **llama.cpp `llama-embedding`** (같은 ggml, 같은 GGUF) | *구현* 정확도 — "같은 그래프를 같게 짰는가" | max abs diff ≤ 1e-4, cosine ≥ 0.99999 |

sentence-transformers는 fp32 PyTorch이므로 우리 F16/Q8_0 결과와 정확히 같지 않음 — 의미적 oracle. llama.cpp는 같은 ggml 커널과 같은 GGUF를 쓰므로 거의 비트 단위 일치 — 구현 디버깅에 강력. 둘 다 통과해야 진짜 정확.

**골든 벡터 생성** (`tools/dump_golden.py`, 두 모드)
```
dump_golden.py --ref sentence-transformers \
    --model BAAI/bge-small-en-v1.5 --corpus tests/corpus/eval.txt \
    --out tests/fixtures/golden/st-bge-small-f16.bin

dump_golden.py --ref llama-cpp \
    --model models/bge-small-en-v1.5-q8_0.gguf --corpus tests/corpus/eval.txt \
    --out tests/fixtures/golden/llama-bge-small-q8.bin
```
- llama.cpp는 *생성 시점만* 의존 (시스템에 빌드된 `llama-embedding` 바이너리 호출). CI / 런타임 의존 아님.
- 생성된 .bin은 git에 commit (~수십 KB).

**테스트 코퍼스** — 직접 작성하지 않고 기존 오픈 데이터셋 활용
- 영어 BERT 검증: HuggingFace `mteb/stsbenchmark-sts` (Apache 2.0)에서 100문장 샘플링 → `tests/corpus/eval.txt` (~30KB)
- 다국어 모델 도입 시 MTEB 다국어 트랙에서 같은 패턴
- MTEB / sentence-transformers / llama.cpp 모두 MIT/Apache. 비교 알고리즘만 자체 작성 (50줄, 외부 의존 추가 가치 없음)

**테스트 (`tests/integration/golden_test.cpp`)**
1. 모델 + 코퍼스 + 골든 .bin 로드
2. NanoEmbed로 동일 입력 임베딩
3. 입력별 cosine + max abs diff 계산
4. Oracle별 임계 어서션:
   - vs sentence-transformers: min cosine ≥ 0.9999, mean cosine ≥ 0.99999
   - vs llama.cpp: max abs diff ≤ 1e-4, min cosine ≥ 0.99999

**§9 레이어 격리 정합성과의 관계**
- 골든 = *최종 임베딩만* — end-to-end 검증.
- 레이어 격리 = *N번째 인코더 블록 입출력* — 구간 디버깅.
- 골든이 깨지면 레이어 격리 테스트로 어느 블록부터 깨졌는지 좁힘.

---

## 디렉터리 레이아웃 (아키텍처가 그대로 매핑)

```
NanoEmbed/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── cmake/
├── third_party/ggml/                     # submodule (M1)
├── include/nanoembed/                    # 안정 공개 헤더
│   ├── nanoembed.h                       # C ABI (M1: 시그니처, M3+: 채워짐)
│   └── nanoembed.hpp                     # C++ RAII wrapper (M7)
├── src/
│   ├── api/c_api.{cpp}                   # M1 stub → M3+ 구현
│   ├── gguf_scanner.{h,cpp}              # M2
│   ├── tokenizer/wordpiece.{h,cpp}       # M3
│   ├── model/
│   │   ├── bert_config.{h,cpp}           # M3
│   │   ├── layer_loader.{h,cpp}          # M4
│   │   └── activation_store.{h,cpp}      # M5
│   ├── forward/{embed_layer,attention,ffn,pool}.{h,cpp}
│   ├── runtime/
│   │   ├── streaming_runner.{h,cpp}      # M4
│   │   ├── batched_runner.{h,cpp}        # M5
│   │   └── compression/                  # M6
│   └── embedder.{h,cpp}
├── tools/
│   ├── nanoembed-inspect/                # M1·M2 — GGUF 메타/레이어 맵
│   ├── nanoembed-cli/                    # M3 — stdin → 임베딩 벡터
│   ├── nanoembed-bench/                  # M3+ — 시나리오 1개 → JSON
│   ├── dump_golden.py                    # M3 — sentence-transformers / llama.cpp로 골든 생성
│   └── dump_hf_activations.py            # M3 — 레이어 격리 픽스처 생성
├── bench/
│   ├── scenarios.yaml                    # 시나리오 정의
│   ├── runner.py                         # 전체 슈트 실행
│   ├── compare.py                        # baseline 대비 회귀 검출
│   ├── baseline/M{N}.json                # 마일스톤별 baseline (commit)
│   └── results/                          # 실행별 결과 (gitignore)
├── tests/
│   ├── corpus/eval.txt                   # STS-B 100문장 샘플
│   ├── fixtures/
│   │   ├── golden/                       # 골든 .bin (commit)
│   │   └── activations/                  # 레이어 격리 .npy (commit)
│   ├── unit/
│   └── integration/golden_test.cpp
└── bindings/node/                        # M8
```

---

## 마일스톤

각 마일스톤은 **혼자서도 돌아가고 검증되는 산출물**을 가집니다. M3 출력이 이후 모든 마일스톤의 정합성 oracle입니다.

### M1 — Skeleton + ggml 벤더링 + 공개 API 동결 + GGUF smoke test
**목표** 빌드 가능, ggml 링크, **공개 C ABI 시그니처 전부 link 가능 (impl은 stub)**, GGUF 메타 출력.

**산출물**
- `CMakeLists.txt`, `third_party/ggml` submodule
- `include/nanoembed/nanoembed.h` — 「공개 API」섹션의 시그니처 전부
- `src/api/c_api.cpp` — 모든 함수 stub (`return NANOEMBED_ERR_INVALID_ARG;` 또는 `nullptr`)
- `tools/nanoembed-inspect`: GGUF 경로 → 버전·텐서 수·메타 KV 출력
- macOS·Linux 빌드 스크립트 / GitHub Actions
- `LICENSE`, `.gitignore`, `cmake/` 헬퍼

**검증**
- `nanoembed-inspect bge-small-en-v1.5.gguf` 가 메타 정상 출력
- 별도 디렉터리에서 `#include <nanoembed/nanoembed.h>` 한 C 프로그램이 모든 심볼에 링크 성공 (실행은 에러 코드 반환)
- `-Wall -Wextra -Wpedantic` 클린 빌드

**리스크**
- ggml CMake 타겟 이름 / Apple Silicon에서 빌드되는 commit pin 확인 필요
- API 시그니처를 너무 일찍 동결해 후속 마일스톤에서 흔들릴 위험 → 「공개 API」섹션의 마일스톤 매트릭스로 이미 채울 자리 검증됨

---

### M2 — GGUF 스캐너: BERT 레이아웃 검증
**목표** ggml의 평탄한 텐서 리스트를 BERT 인코더의 타입 있는 `ModelManifest`로 변환.

**산출물**
- `src/gguf_scanner.{h,cpp}`
  - `ModelManifest scan(const std::string& path)`
  - `struct LayerSlot { TensorRef q, k, v, o, ffn_up, ffn_down, ln1, ln2; };`
  - `struct ModelManifest { ArchInfo arch; vector<LayerSlot> layers; TensorRef tok_embed, pos_embed, type_embed, ln_embed, pooler?; };`
- `general.architecture == "bert"` 검증
- `nanoembed-inspect`에 레이어 맵 출력 추가
- 픽스처 GGUF 기반 단위 테스트

**검증**
- bge-small-en-v1.5에 대해 `layers=12, hidden=384, heads=12, vocab=30522` 출력
- 텐서 누락 시 forward 단계가 아니라 스캔 단계에서 명확히 에러

**리스크**
- GGUF 메타키 명세가 컨버터마다 다름 → llama.cpp `convert-hf-to-gguf.py` 결과를 source of truth로 고정하고 문서화

---

### M3 — Naive in-memory 임베더 + **벤치마크/골든 슈트 첫 가동**
**목표** End-to-end forward. 메모리 예산 무시하고 전부 로드. **이 마일스톤이 정확도/성능 baseline의 origin** — 이후 모든 마일스톤이 여기서 commit한 baseline JSON·골든 .bin을 oracle로 사용.

**산출물 (구현)**
- `src/tokenizer/wordpiece.{h,cpp}` — GGUF 메타에서 vocab 로드
- `src/forward/{embed_layer,attention,ffn,pool}.{h,cpp}`
- `src/embedder.{h,cpp}` + `src/api/c_api.cpp` 구현 (M1 stub 채움)
- `tools/nanoembed-cli`: `echo "hello" | nanoembed-cli model.gguf`

**산출물 (벤치마크 인프라)**
- `tools/nanoembed-bench` — RSS·latency·page faults 측정 + JSON 출력
- `bench/scenarios.yaml` 에 `single_short_f16` 등록
- `bench/runner.py`, `bench/compare.py`
- `bench/baseline/M3.json` commit

**산출물 (정확도 인프라)**
- `tools/dump_golden.py --ref sentence-transformers` 모드
- `tools/dump_hf_activations.py` — 레이어 격리 픽스처 생성기
- `tests/corpus/eval.txt` (STS-B 100문장 샘플)
- `tests/fixtures/golden/st-bge-small-f16.bin` commit
- `tests/integration/golden_test.cpp`

**검증**
- 토크나이저 단독: HF tokenizer와 토큰 ID 완전 일치
- 골든 vs sentence-transformers (F16): min cosine ≥ 0.9999, mean ≥ 0.99999
- 레이어 격리 정합성: 12개 인코더 블록 모두 통과
- `nanoembed-bench` 가 `single_short_f16` 시나리오에서 모든 지표 JSON 출력
- F32 GGUF가 컨버터에서 쉽게 나온다면 F32에서도 동일 임계 통과 (없으면 future work)

**리스크**
- F16 + reduction order 차이로 인한 수치 drift → 레이어별 tolerance를 경험적으로 설정. 레이어 격리 테스트가 첫 깨지는 블록 식별
- WordPiece edge case → 토크나이저를 먼저 통과시킨 뒤 골든·격리 픽스처 생성

---

### M3.5 — 그래프 아레나 정리 (M4 진입 선행)
**목표** `embed()`가 호출마다 고정 256 MiB 아레나를 잡고 버리던 구조를 ggml 그래프 할당기로 교체. **수치는 M3와 동일**(골든이 oracle)하고, 메모리·크래시만 고친다.

M4에 합치지 않고 분리한 이유는 **측정 가능성**이다. M3 peak RSS 330 MiB의 78%가 그래프 아레나라, 이 상태로 스트리밍을 넣으면 가중치 절감분이 아레나 노이즈에 묻혀 M4가 자기 효과를 증명하지 못한다.

**M3 baseline 해부 (실측)**

| 구성 | 크기 | 실사용 |
|---|---|---|
| 프로세스 바닥 | 4.15 MiB | — |
| 모델 가중치 | ~64.2 MiB | 전부 |
| 토크나이저 + 메타 | ~5.7 MiB | 전부 |
| **그래프 아레나** | **256.0 MiB** | **12.3 MiB (4.8%)** |
| **peak** | **330.05 MiB** | |

원인은 `ggml_context`가 bump allocator라 12개 블록의 중간 텐서를 **재사용 없이** 전부 살려두는 것. 게다가 `std::vector<uint8_t>`의 값 초기화가 256 MiB 전체를 매 호출 memset해 안 쓰는 95%까지 상주시킨다. 같은 이유로 S≳270에서 아레나가 넘쳐 `GGML_ASSERT`로 **abort** — 기본 `max_seq_len`이 512이므로 이건 옵트인 엣지케이스가 아니라 기본 설정에서 닿는 경로였다.

**산출물 (구현)**
- `src/embedder.{h,cpp}` — `ggml_graph_compute_with_ctx` + `no_alloc=false` 아레나 → `ggml_backend_t`(CPU) + `ggml_gallocr` 조합으로 전환:
  - 호출마다 여는 컨텍스트는 `no_alloc=true`인 **메타 컨텍스트**(텐서 구조체 전용, 수백 KB). 텐서 데이터는 galloc이 소유한 단일 버퍼에서 수명 분석 기반으로 배치·재사용
  - 입력 텐서는 `ggml_set_input`으로 표시하고 **할당 이후** `ggml_backend_tensor_set`으로 주입 (`alloc_graph` 전에는 `->data`가 없음)
  - 버퍼는 핸들 수명 동안 유지 → 호출당 재폴트 소멸
- **생성 시점 worst-case 예약** — 생성자에서 `max_seq_len` 기준 `ggml_gallocr_reserve`. 부담 불가한 모델은 추론 도중이 아니라 로드 시점에 실패하고(§6 에러 모델), peak RSS가 입력 길이와 무관하게 확정된다. `size_max >= node_size` 비교 덕에 이후 짧은 입력은 재할당 없이 이 예약을 재사용한다
- `abort()` 제거 — 예약/할당 실패를 예외로 올려 C ABI 경계에서 `nanoembed_status_t`로 변환
- `nanoembed-inspect --graph` — 예약된 활성값 버퍼 크기 보고 (C ABI는 M1에서 동결이므로 진단 도구 쪽에 노출)

**산출물 (테스트/벤치)**
- `tests/integration/seq_len_test.cpp` — 기본 파라미터로 (1) 512토큰 입력 성공, (2) 초과 입력은 truncate, (3) 장문/단문을 교차 반복해도 재사용 버퍼가 결과를 오염시키지 않음(cosine ≥ 0.99999)
- `tests/corpus/eval_long.txt` + `bench/scenarios.yaml`의 `single_long_f16` (S=512 포화) — **M3에서는 실행 자체가 불가능했던** 시나리오라 M3 baseline이 없고 M3.5부터 시작
- `bench/baseline/M3.5.json` commit

**검증**
- 골든 파리티 **무변경** 통과 (min cosine ≥ 0.9999, mean ≥ 0.99999) — "수치를 안 바꿨다"의 유일한 증거
- 레이어 격리 12블록 통과, `ctest` 7/7
- `nanoembed-inspect --graph`: max_seq_len=512 worst case **15.76 MiB** (M3의 고정 256 MiB 대비 1/16, 옛 방식이 S=512에서 *필요로 했을* ~600 MiB 대비 1/38)
- `compare.py` (M3 baseline 대비, 동일 머신) — `single_short_f16`:

  | 지표 | M3 | M3.5 | |
  |---|---|---|---|
  | `rss_peak_lifetime_mb` | 330.05 | **76.35** | −76.9% |
  | `pss_peak_sampled_mb` | 326.9 | 73.13 | −77.6% |
  | `uss_peak_sampled_mb` | 325.6 | 71.85 | −77.9% |
  | `page_faults_minor` | 327,685,001 | **0** | 아레나 재폴트 소멸 |
  | `latency_p50_ms` | 240.05 | **41.97** | −82.5% |
  | `throughput_items_per_sec` | 4.00 | 20.27 | +407% |

  peak == baseline == 76.35 MiB로 **측정 창 안에서 메모리가 전혀 자라지 않는다**. 지연 5.7배 개선은 알고리즘이 아니라 호출마다 256 MiB를 memset하던 비용이 사라진 것.
- 신규 `single_long_f16` (S=512 포화, M3에는 대응 baseline 없음): peak **93.01 MiB**, minor fault 1, p50 2084 ms. 예약은 가상이고 상주는 실사용을 따르므로, 짧은 입력은 15.76 MiB 예약분 중 ~2 MiB만 상주한다 — worst-case 예약이 단문 RSS를 손해보지 않는 이유
- 부수 수정: `compare.py`가 지연 **개선**을 "REGRESSION"으로 라벨링하던 문제 — 양방향 감지는 유지하되 IMPROVEMENTS로 분리하고 `--strict`는 악화에만 실패

**리스크**
- **가중치 텐서의 버퍼 소속** — 가중치는 여전히 평범한 `ggml_context`(`gguf_init_from_file(no_alloc=false)`)에 있다. CPU 백엔드는 호스트 포인터를 그대로 읽고 gallocr은 `->data`가 이미 있는 텐서를 건너뛰므로 그대로 동작함을 확인했다. 다른 백엔드를 붙이거나 ggml을 올릴 때 "모든 텐서는 backend buffer 소속" 어서션이 들어오면 `ggml_backend_alloc_ctx_tensors` 경로로 옮겨야 한다
- 입력 주입 순서를 되돌리면 널 참조 — `seq_len_test`와 골든이 회귀를 잡는다

**범위 밖 (M4)** 레이어 스트리밍, 양자화 가중치, 배치. **M3.5 이후에도 peak은 ~74 MiB 밑으로 못 내려간다** — 그중 64 MiB가 가중치이고 그건 스트리밍이 푸는 문제다. M4의 "peak RSS < 40MB"는 M3.5(아레나) + M4(스트리밍+양자화)가 **둘 다** 있어야 도달한다.

---

### M3.6 — 모델 교체 가능 구조 + eurobert(jina-v5-nano) 지원
**목표** GGUF 모델 패밀리를 갈아끼울 수 있게 만들고, 두 번째 패밀리로 `jina-embeddings-v5-text-nano-retrieval`(arch `eurobert`)을 붙인다. **M4 스트리밍의 메모리 절감을 잴 대상 모델이 이것**이므로 M4 선행이다 — bge-small은 가중치가 64 MiB뿐이라 스트리밍 효과를 과소평가한다.

**두 모델이 실제로 얼마나 다른가** (`nanoembed-inspect`로 확인)

| | bge-small (`bert`) | jina-v5-nano (`eurobert`) |
|---|---|---|
| 위치 인코딩 | 학습된 `position_embd` | **RoPE** (`rope.freq_base=1e6`) |
| 정규화 | LayerNorm (weight+bias) | **RMSNorm** (weight만) |
| FFN | `up`/`down` + GELU | **`gate`/`up`/`down` (SwiGLU)** |
| attention bias | q/k/v/o 전부 | **없음** |
| segment embedding | `token_types` | **없음** |
| 토크나이저 | WordPiece 30,522 | **byte-level BPE 128,256 + merges 280,147** (`pre="jina-v5-nano"`) |
| pooling | CLS | `pooling_type=3` (**LAST**) |
| context | 512 | 8,192 |
| 파라미터 | 33M | 212M |

겹치는 텐서가 사실상 없다 — `token_embd.weight` 하나뿐이고 블록은 `attn_norm`/`ffn_gate`/`ffn_up`/`ffn_down`/`ffn_norm` 구성. **모델 경로 교체가 아니라 새 패밀리 구현**이다.

**산출물 (구조 — 완료)**
- `src/arch/model_arch.h` — `ModelArch` 인터페이스(`params`/`inputs`/`default_pooling`/`bind_weights`/`build_graph`) + `create_model_arch()` 레지스트리 (`general.architecture` 디스패치)
- `src/arch/bert_arch.{h,cpp}` — 기존 BERT 경로를 인터페이스 뒤로 이동. `scan_gguf`/`ModelManifest`(M2 산출물)는 **BERT의 사적 디테일로 강등** — 공유 계약이 아님
- `src/tokenizer/tokenizer.h` + `registry.cpp` — `Tokenizer` 인터페이스 + `tokenizer.ggml.model` 디스패치. 아키텍처와 **독립적으로** 변하므로 별도 seam
- `forward::PoolType::Last` 추가
- `bench/model_source.py` — 시나리오의 `model:`이 `hf:<repo>:<file.gguf>` 형식을 받는다. **캐시에서만 해석하고 절대 자동 다운로드하지 않는다** (벤치가 조용히 다른 quant을 재는 사고 방지)
- **활성값 예약 시점 이동** — 생성자에서 모델 context 전체를 예약하던 것을 `Embedder::reserve(max_seq_len)`로 옮겨 context 생성 시 호출. eurobert의 8192 context를 그대로 예약하면 attention이 O(S²)이라 12 heads × 8192² × 4B ≈ **3.2 GB**가 된다. 기본 캡은 512, 더 원하는 호출자는 명시하고 지불

**산출물 (eurobert — 미구현)**
- `src/arch/eurobert_arch.{h,cpp}` — RoPE + RMSNorm + SwiGLU + bias 없는 projection. **ggml 내장으로 대부분 해결**(`ggml_rope_ext`, `ggml_rms_norm`, `ggml_silu`)이라 forward 자체는 블록 하나 분량
- `src/tokenizer/bpe.{h,cpp}` — byte-level BPE. **이 마일스톤 비용의 대부분**. `tokenizer.ggml.merges`(280k) 로드, byte-level 매핑, `pre="jina-v5-nano"` pre-tokenizer 정규식
- `tests/fixtures/tokenizer/jina-v5-nano-eval.tsv` — HF 토큰 ID 픽스처 (BPE 정확도 oracle)
- `tests/fixtures/golden/st-jina-v5-nano.bin` — 골든 임베딩 픽스처
- 공개 ABI에 `NANOEMBED_POOL_LAST` 추가 (M1 동결 정책상 **추가는 허용**)

**검증**
- BERT 경로 **무변경**: 골든/레이어격리/토크나이저 픽스처 전부 그대로 통과 (구조 리팩터링 시점에 확인 완료)
- eurobert 토크나이저: HF tokenizer와 토큰 ID 완전 일치
- eurobert 골든: sentence-transformers 대비 min cosine ≥ 0.9999, mean ≥ 0.99999
- **Q3_K_M 동작 확인** — `ggml_mul_mat`/`ggml_get_rows`가 K-quant를 네이티브 처리하므로 forward 수정 없이 될 가능성이 높지만 **가정이며 검증 대상**. F16 대비 정확도 열화폭도 같이 기록
- `bench/scenarios.yaml`의 `single_short_jina_q3km` 실행 → `bench/baseline/M3.6.json`

**리스크**
- **BPE 토크나이저가 실제 비용의 대부분**이고 정확도 오라클(HF 픽스처)을 새로 떠야 한다. WordPiece와 공유할 코드가 없다
- Q3_K_M이 그냥 되지 않으면(예: `get_rows`가 특정 K-quant를 미지원) F16/Q8_0으로 먼저 정합성을 잡고 quant을 분리 처리
- `pooling_type=3`의 LAST가 padding 없는 B=1 경로 가정에 의존 — M5 배치에서 mask-aware로 다시 봐야 함
- eurobert의 8192 context에서 장문 벤치를 돌리면 활성값이 지배적이 된다. M4 스트리밍은 **가중치**를 줄이는 것이므로, 절감폭 측정은 예약 캡을 고정한 채 비교해야 의미가 있다

---

### M4 — 레이어 스트리밍 (헤드라인 기능) + 양자화 가중치 지원
**목표** M3의 "전부 로드"를 **레이어별 load → compute → free**로 교체. 단일 입력 모드. 출력은 M3와 동일 tolerance, RSS 급감. **여기서 Q8_0/Q4_K_M 양자화 가중치 도입** (스트리밍의 핵심 가치는 양자화 모델에서 두드러지므로 같이 묶음).

**스트리밍 inner loop (의사 코드)**
```
mmap(gguf_path)
load_embedding_tables_into_persistent_ctx()    // 항상 상주
acts = embed_tokens(input_ids)                  // [seq_len, hidden]

for layer_idx in 0..num_layers:
    transient_ctx = ggml_init(small_buffer)
    layer_w = layer_loader.load(transient_ctx, layer_idx)  // ggml view 또는 복사 (M4에서 결정)
    graph = build_encoder_block(transient_ctx, acts, layer_w)
    ggml_graph_compute(graph, threadpool)
    acts = copy_out(graph.output)               // activation buffer로
    ggml_free(transient_ctx)                    // 다음 레이어 로드 전에 가중치 해제
    madvise(layer_region, MADV_DONTNEED)

return l2_normalize(pool(acts, attn_mask))
```
**불변식**: 어느 시점이든 메모리에는 *embedding tables + activations + 단 하나의 레이어 가중치* 만 존재.

**산출물 (구현)**
- `src/model/layer_loader.{h,cpp}` — 레이어 인덱스 → transient ctx에서 사용 가능한 `LayerWeights`
- `src/runtime/streaming_runner.{h,cpp}` — 레이어 루프 + persistent/transient 구분
- Q8_0, Q4_K_M 가중치 경로 (ggml 커널 이용)
- `nanoembed-cli`에 `--streaming` 플래그, 큰 모델은 디폴트 ON

**산출물 (테스트)**
- `dump_golden.py --ref llama-cpp` 모드 — Q8_0 GGUF에 대해 llama.cpp `llama-embedding` 호출
- `tests/fixtures/golden/llama-bge-small-q8.bin` commit
- `golden_test.cpp` 에 llama.cpp oracle 케이스 추가
- `bench/scenarios.yaml` 에 `single_short_q8_streaming` 등록
- `bench/baseline/M4.json` commit

**검증**
- 정확도 1: 출력 vs M3 (둘 다 F16) cosine ≥ 0.9999 — 스트리밍이 수학을 안 바꿈
- 정확도 2: 출력 vs llama.cpp (Q8_0) max abs diff ≤ 1e-4, min cosine ≥ 0.99999 — 우리 ggml 그래프가 llama.cpp와 같음을 확인
- 메모리: bge-small Q8_0 단일 추론 peak RSS < **40 MB**, M3 동일 입력 대비 RSS 감소율 ≥ 50%
- 안정성: 10k 연속 호출 후 fd / VM region leak 없음
- 성능: `bench/compare.py M3.json M4.json` 의 latency 회귀 보고서 — single_short은 늘어남 (예상), page faults 증가도 예상. 의도 외 회귀 시 fail
- (스트레치) 더 큰 모델 일반화는 후속 마일스톤에서 모델 도입 시점에 측정

**리스크**
- macOS·Linux의 cold mmap 페이지 evict 동작 차이 → `MADV_DONTNEED` 적용 후 RSS 실측으로 검증
- 단일 호출 latency가 M3 대비 크게 증가 (가중치를 매번 다시 읽으므로) → M5 배치에서 분할상환

---

### M5 — 레이어 단위 배치 임베더
**목표** 스트리밍 I/O 비용을 N개 입력에 분할상환. 한 레이어 윈도우 안에서 배치 전체를 흘려보냄. **이 단계부터 활성값 메모리가 dominant**가 됨 → 계측 필수.

**산출물 (구현)**
- `src/model/activation_store.{h,cpp}` — `[B, seq_len, hidden]` + attention mask, 예산 초과 시 sub-batch 자동 분할
- `src/runtime/batched_runner.{h,cpp}` — 외부 루프는 streaming runner와 동일, GraphBuilder에 batch 차원 텐서 전달
- 길이 버킷팅 + 패딩 (입력을 길이 순 정렬 후 버킷별 패딩) — 임베딩 입력의 길이 분산 특성 활용
- `nanoembed_embed_batch` C ABI가 진짜 배치로 동작 (M3에서는 N회 단일 호출이었음)

**산출물 (테스트)**
- `bench/scenarios.yaml` 에 `batch32_mixed_q8`, `batch128_mixed_q8` 등록
- `golden_test.cpp` 에 배치 케이스 추가 — 같은 입력 단일 vs 배치 출력 일치 확인
- `bench/baseline/M5.json` commit

**검증**
- 정확도: batched 출력 = 단일 streaming 출력 (1e-4 이내, 같은 입력)
- 정확도: 배치 모드 vs sentence-transformers 골든 — 단일과 같은 임계 통과
- 성능: bge-small Q8_0, 배치 32, 시퀀스 128 기준 **per-item amortized latency ≤ M4 단일 호출 latency / 10** — 분할상환이 의도대로 작동
- 메모리: instrumentation으로 RSS 구성 비율 출력 → activation store 비중이 dominant로 부상하는지 확인 (M6 진입 판단 근거)
- `compare.py M4.json M5.json`: per-item 기준 page faults 감소, throughput 증가, peak RSS 단조 비증가 확인
- 절대 RSS 한계는 모델·배치·시퀀스 파라미터 의존이라 본 마일스톤에서는 *비율* 어서션, *절대값* 은 ENV로 세팅

**리스크**
- 가변 길이 입력 → 단순 패딩은 compute 낭비. 버킷팅으로 시작, packing은 future work
- 자동 sub-batch 트리거 정책: 명시 config + 자동 정책 둘 다 노출

---

### M6 — 활성값 압축 (TurboQuant 등) **[후순위]**
**목표** 인터-레이어 활성값을 int8/int4로 압축해 더 큰 배치 수용. **M5의 activation store가 실제 병목임을 계측으로 확인한 후에만 진행**.

**산출물**
- `src/runtime/compression/turboquant.{h,cpp}` — 레이어 경계에서 quantize/dequantize, per-row scale
- activation store에 "compressed mode" (int8 + scales)
- batched_runner: 레이어 출력 직후 quant, 다음 레이어 입력 직전 dequant
- config: `compression: none | int8 | int4`
- STS-B 부분 셋으로 품질 회귀 테스트

**검증**
- 동일 RSS 예산에서 batch 64 → batch 128 수용
- STS-B Spearman 상관 -0.01 이내
- round-trip 오버헤드 < 15%

**리스크**
- 정확한 알고리즘 (TurboQuant vs vanilla per-row absmax) 확정 필요
- 활성값 outlier 처리 (per-row이 통상 충분, 안되면 per-token)

---

### M7 — 패키징 + C++ 래퍼 + 설치
**목표** ABI는 M1에서 이미 동결됨. 이 마일스톤은 *배포* 차원: install / find_package / shared lib / RAII wrapper.

**산출물**
- `include/nanoembed/nanoembed.hpp` — RAII C++ wrapper (~30줄)
- shared lib 옵션 (`BUILD_SHARED_LIBS=ON` 시 `libnanoembed.dylib`/`.so`)
- `cmake --install` + `nanoembed-config.cmake` (find_package 지원)
- pkg-config 파일, SOVERSION
- 별도 디렉터리의 순수 C 컨슈머 ABI 회귀 테스트 (M1의 stub 링크 테스트의 확장 — 이번엔 실제 임베딩 호출)

**검증**
- 별도 테스트 프로젝트에서 `find_package(NanoEmbed)` + 순수 C 예제 프로그램이 NanoEmbed CLI와 동일 임베딩 출력
- shared lib SOVERSION 정상

---

### M8 — Node.js 바인딩
**목표** `npm install nanoembed` 후 JS에서 호출.

**산출물**
- `bindings/node/`: N-API + node-addon-api, `package.json`, CMake-js 또는 binding.gyp
- `Embedder` 클래스: `embed(string): Float32Array`, `embedBatch(string[]): Promise<Float32Array[]>`
- 비동기 변형으로 event loop 비차단
- macOS arm64 / Linux x86_64·arm64 prebuild 또는 source 빌드 fallback

**검증**
- `node example.js` 결과 = C++ CLI 출력
- 호출당 event loop 차단 < 10ms (병렬 `setInterval`로 측정)

---

## 마일스톤 진입 게이트

각 마일스톤은 다음 4개를 모두 통과해야 다음 마일스톤 진입 — `bench/baseline/M{N}.json` 과 `tests/fixtures/golden/*.bin` 이 다음 마일스톤의 oracle.

1. **단위 테스트** — `ctest` 통과
2. **정확도** — `golden_test.cpp` 가 해당 마일스톤이 책임지는 oracle 임계 충족 (M3: sentence-transformers / M4+: + llama.cpp)
3. **벤치마크** — `bench/runner.py && bench/compare.py baseline/M{N-1}.json results/<sha>.json` 가 회귀 없음으로 종료
4. **CLI 스모크** — `nanoembed-cli` 로 한 번 임베딩 출력

수동 검증:
```
cmake -B build && cmake --build build && ctest --test-dir build --output-on-failure
./build/tools/nanoembed-cli models/bge-small-en-v1.5-q8_0.gguf <<< "hello world"
python bench/runner.py --model models/bge-small-en-v1.5-q8_0.gguf
python bench/compare.py bench/baseline/M{N-1}.json bench/results/$(git rev-parse --short HEAD).json
```

---

## 라이선스 / 미해결

- NanoEmbed 자체 라이선스 미정 (ggml은 MIT). M1 시작 전에 결정 권장.
- TurboQuant 알고리즘 세부 사양은 M6 진입 시점에 확정 (vanilla per-row absmax int8을 디폴트 후보로).
- 두 번째 타겟 모델 선정 시점에 M4·M5 RSS 한계 절대값 추가.
