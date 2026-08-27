# NanoEmbed

NanoEmbed는 텍스트를 의미 벡터로 바꾸는 C++17 라이브러리다. 서버뿐 아니라 메모리가 제한된 엣지 장치에서도 임베딩 모델을 실행하는 것이 목표다.

임베딩은 문장 하나를 고정 길이 실수 배열로 표현한 값이다. 의미가 비슷한 문장은 가까운 벡터가 되므로 검색, 추천, 중복 탐지, RAG의 검색 단계에 사용할 수 있다.

## 현재 구현 범위

두 가지 모델 계열을 실행할 수 있고, 두 계열 모두 Hugging Face 결과와 대조해 검증되어 있다.

| 모델 구조 | 대표 모델 | 토크나이저 | 풀링 | 상태 |
|---|---|---|---|---|
| `bert` | `bge-small-en-v1.5` | `bert` WordPiece | CLS | 지원 |
| `gemma3` | `microsoft/harrier-oss-v1-270m` | `llama` SentencePiece-BPE | LAST | 지원 |
| `eurobert` | `jina-embeddings-v5-text-nano` | `gpt2` byte-level BPE | — | 식별만, 미구현 |

두 모델은 공통점이 거의 없다. 학습된 위치 임베딩 대 RoPE, LayerNorm 대 RMSNorm,
GELU MLP 대 게이트형 GeGLU, 편향 유무, 양방향 대 causal 어텐션, 그리고
`gemma3`는 쿼리 헤드 4개가 KV 헤드 하나를 공유한다. 그래서 모델 계열 경계가
개별 연산 빌더보다 위에 있다.

- C API로 단일 문장과 여러 문장을 임베딩할 수 있다. 여러 문장 API는 아직 내부에서 한 문장씩 처리한다.
- 풀링 기본값은 **모델이 학습된 방식**을 따른다. 마지막 토큰으로 풀링하는 모델에
  평균 풀링을 적용하면 겉보기에 멀쩡한 잘못된 벡터가 나오는데, 어느 쪽이 맞는지는
  호출자가 아니라 모델의 성질이기 때문이다.
- 중간 계산값을 저장하는 버퍼는 호출마다 새로 만들지 않고 컨텍스트별로 재사용한다.
- Linux에서는 context의 `use_streaming=1`로 read-only GGUF mmap과 레이어별
  residency 제어를 선택할 수 있다. 기본값 `0`은 기존 eager 경로다.

모델 핸들은 처음에는 메타데이터만 읽는다. 최초로 성공한 context가 해당 모델 핸들의
실행 모드를 eager 또는 streaming으로 고정하며, 이후에는 같은 모드 context만 만들 수
있다. 값은 정확히 0 또는 1이어야 하고, mixed mode, 비-Linux streaming, 초기화 실패는
eager로 대체하지 않고 오류를 반환한다. A/B 비교에는 같은 파일을 가리키는 별도 모델
핸들을 사용해야 한다.

정확도는 sentence-transformers와의 코사인 유사도로 검증한다. B5 저장 결과에서
`bge-small`의 최악 코사인은 0.999999156, F32 `harrier-270m`은 0.999999725다.
후자의 코퍼스에는 한국어·일본어·중국어·키릴·아랍어·이모지가 포함된다.

양자화된 GGUF도 그대로 읽는다. `harrier-270m`을 같은 기준값과 비교하면
q8_0은 평균 0.999760(최악 0.999106), q4_k는 평균 0.984727(최악 0.947608)다.
평균이 최악의 경우를 가린다는 점에 주의한다. 자세한 수치와 해석은
[마일스톤 해설서 7.8](docs/milestones-explained.ko.md)에 있다.

전체 계획은 [PLAN.md](PLAN.md)에서 볼 수 있다. 각 마일스톤의 용어,
필요성, 대안, 선택 이유와 구체적인 구현은
[마일스톤 해설서](docs/milestones-explained.ko.md)에서 자세히 설명한다.

## 처음 읽는다면

GGML, GGUF, 트랜스포머가 익숙하지 않아도 코드를 읽을 수 있도록 별도 가이드를 제공한다.

1. [프로젝트와 문서 구성](docs/guide/26080401/00-index.md)
2. [프로젝트 전체 흐름](docs/guide/26080401/01-overview.md)
3. [코드를 읽기 위한 GGML 기초](docs/guide/26080401/02-ggml-primer.md)
4. [BERT 임베딩 모델의 계산 과정](docs/guide/26080401/03-model-theory.md)

그다음에는 모델 로드, 순전파, API와 소유권, 테스트 순서로 실제 코드를 따라갈 수 있다.

## 빌드와 테스트

GGML은 Git 서브모듈로 포함되어 있다. 처음 받은 저장소라면 서브모듈도 함께 초기화한다.

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

모델 파일이 없으면 그 모델이 필요한 테스트는 건너뛴다. 전체 정확도·양자화 회귀
테스트를 실행하려면 아래 세 gate 대상 파일을 모두 둔다. Q4 정확도 보고까지 만들려면
네 번째 파일도 선택적으로 둔다.

```text
models/bge-small-en-v1.5-f16.gguf
models/harrier-270m.gguf
models/harrier-270m-q8_0.gguf
models/harrier-270m-q4_k.gguf  # optional, report-only
```

```sh
curl -fL -o models/bge-small-en-v1.5-f16.gguf \
  https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f16.gguf
curl -fL -o models/harrier-270m.gguf \
  https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m.gguf
curl -fL -o models/harrier-270m-q8_0.gguf \
  https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m-q8_0.gguf
```

경로를 바꾸려면 `NANOEMBED_TEST_MODEL`, `NANOEMBED_TEST_MODEL_GEMMA3`,
`NANOEMBED_TEST_MODEL_GEMMA3_Q8` 캐시 변수를 CMake에 넘긴다. Q8_0은 같은 F32
기준값에 대해 문장별 0.9985, 평균 0.9995 이상의 별도 회귀 게이트를 적용한다.
Q4 경로는 `NANOEMBED_TEST_MODEL_GEMMA3_Q4`이며 새 gate 없이 report-only다.

CI에서는 모델을 내려받은 뒤 `NANOEMBED_REQUIRE_MODEL=ON`으로 설정한다. 모델이 없어서 테스트가 조용히 건너뛰는 일을 막기 위한 설정이다.

## CLI 사용

표준 입력으로 문장을 전달하면 쉼표로 구분된 임베딩 벡터를 출력한다.

```sh
echo "hello world" \
  | ./build/bin/nanoembed-cli models/bge-small-en-v1.5-f16.gguf
```

풀링은 기본적으로 모델이 학습된 방식을 따르고, `--mean` / `--cls` / `--last`로
바꿀 수 있다.

GGUF의 메타데이터와 텐서 구성을 확인하려면 검사 도구를 사용한다.

```sh
./build/bin/nanoembed-inspect models/bge-small-en-v1.5-f16.gguf
```

`--graph`를 추가하면 모델을 실제로 로드하고, 길이 512 입력을 위해 컨텍스트가 예약한 중간 계산 버퍼의 크기도 출력한다.

```sh
./build/bin/nanoembed-inspect \
  models/bge-small-en-v1.5-f16.gguf --graph
```

## GGUF에서 모델 종류를 선택하는 방법

GGUF는 단순한 가중치 묶음이 아니다. 모델 구조, 토크나이저 종류, 텐서 이름과 자료형 같은 정보도 함께 저장한다.

NanoEmbed는 다음 두 값을 서로 독립적으로 읽는다.

- `general.architecture`: BERT처럼 어떤 계산 구조를 사용할지 결정한다.
- `tokenizer.ggml.model`: WordPiece나 BPE처럼 문자열을 어떤 방식으로 토큰 ID로 바꿀지 결정한다.

지원하지 않는 값을 만나면 다른 구조로 추측해서 실행하지 않는다. GGUF에 실제로 기록된 종류와 아직 구현되지 않은 기능을 오류 메시지로 알린다.

## 벤치마크

벤치마크 도구는 Linux의 `/proc`에서 프로세스 메모리와 페이지 폴트를 읽으므로
`nanoembed-bench`는 Linux에서만 빌드된다. macOS에서는 라이브러리, portable
통계·파서 테스트와 Python 하네스 통합 테스트만 실행되며, CTest는 Linux 전용
`bench_selftest` 두 개를 명시적으로 `Skipped`로 표시한다.

```sh
# Linux에서 모델 없이 측정 도구 자체를 검사한다.
./build/bin/nanoembed-bench --selftest
./build/bin/nanoembed-bench --selftest --memory-profile \
  --memory-profile-interval-ms 10

# 빠른 authoritative warm 성능 측정: 상세 메모리 샘플링은 기본 off다.
.venv/bin/python bench/runner.py \
  --milestone M3.6 \
  --group english_short:20 \
  --selection-seed 0 \
  --cache-state warm \
  --raw-samples-out bench/results/local.samples.json \
  --out bench/results/local-perf.json

# M4 eager/streaming A/B smoke: 모든 paired scenario를 한 artifact에 실행한다.
# N=1은 기능/계약 확인용이며 B5 최종 성능 표본을 대신하지 않는다.
.venv/bin/python bench/runner.py \
  --milestone M4 \
  --samples-per-group 1 \
  --selection-seed 0 \
  --cache-state warm \
  --out bench/results/local-m4-ab-smoke.json

# 최종 메모리 보고용 diagnostic 측정: PSS/USS를 짧은 간격으로 샘플한다.
.venv/bin/python bench/runner.py \
  --milestone M3.6 \
  --group english_short:20 \
  --cache-state warm \
  --memory-profile \
  --memory-profile-interval-ms 10 \
  --out bench/results/local-memory.json

# cold start: 선택된 입력마다 새 native 프로세스와 worker를 만든다.
.venv/bin/python bench/runner.py \
  --milestone M3.6 \
  --filter single_short_f16 \
  --group english_short:5 \
  --cache-state cold \
  --strict-cold \
  --out bench/results/local-cold.json

# 이전 기준값과 비교한다. schema v1(M3/M3.5)도 읽을 수 있다.
.venv/bin/python bench/compare.py \
  bench/baseline/M3.5.json \
  bench/results/local-perf.json \
  --strict
```

위 M4 명령은 eager와 streaming paired scenario를 모두 같은 runner invocation과
artifact에 넣는다. `--filter ..._streaming`만 실행하면 streaming 한쪽만 수집하므로
A/B 결과가 아니다. runner는 각 pair가 execution mode와 이름 외의 controlled field에서
다르면 native process를 시작하기 전에 실패한다.

`--group NAME[:N]`은 반복해서 지정할 수 있다. `:N`이 없으면
`--samples-per-group N`이 적용되고 둘 다 없으면 그룹 전체를 사용한다. 선택은
`--selection-seed`와 입력 내용으로 결정되며 결과에는 선택된 ID와 해시가 남는다.

profile-off는 timed path에서 `smaps_rollup`을 읽거나 sampler thread를 만들지 않는다.
이때도 `statm` 경계 RSS와 커널의 `VmHWM`으로
`rss_peak_lifetime_mb`/`rss_peak_window_mb`를 기록한다. profile-on의 latency는
`diagnostic`이며 PSS/USS의 `*_peak_sampled_mb`는 관측 사이의 순간 피크를 놓칠 수
있는 sampled lower bound다. cold 결과는 `mincore`로 eviction 뒤 resident page가
0인지 기록하며 `--strict-cold`는 검증 실패 시 worker 시작 전에 종료한다.

결과 schema v2에는 p50/p90/p95/p99, population 표준편차와 MAD,
`single_request_items_per_sec`, 요청값/해결값, corpus·모델·binary·Git·build·host
fingerprint가 들어간다. `--raw-samples-out`은 요청별 latency를 별도 sidecar에 쓰고
본문에는 경로·크기·SHA-256만 둔다. 현재 한 번의 runner 실행은
`independent_runs: 1`, `confidence_interval: null`인 기술 통계다. 서로 독립적인 여러
run의 분산이나 신뢰구간을 대신하지 않는다.

M4 결과는 `requested_execution_mode`, `resolved_execution_mode`와 versioned
`execution_mode_resolution` 증거를 기록한다. native worker와 runner는 요청·해결 claim이
다르거나 strict context 생성 증거가 없으면 결과를 만들지 않는다. lazy 모델 로드 이후
cold의 `model_load_ms`는 metadata descriptor 생성이고, eager 전체 가중치 로드 또는
streaming mmap/분류 초기화는 `context_create_ms`에 포함된다. canonical
`startup_to_first_result_ms`는 둘을 모두 포함한다.

성능 수치는 같은 머신과 같은 fingerprint 조건에서 얻은 값끼리 비교해야 한다.
`compare.py --strict`는 핵심 환경 불일치와 기존 latency/RSS gate를 적용한다.
추가 fingerprint 차이는 gate를 바꾸지 않고 해석 전 검토할 진단으로 출력한다.

PyTorch 기준 임베딩 오차는 성능 timed path 밖에서 `golden` CTest가 계산한다.
`build/accuracy-report.json`에는 cosine, 최대/평균 절대 오차, RMSE, norm 오차의
문장별 값과 집계, F32 기준 오차와 별도의 양자화 손실이 기록된다. fixture 옆
provenance manifest와 checksum은 정확한 Hugging Face revision·생성 옵션·패키지
lock·입력/산출물 해시를 검증한다. 기존 fixture는 출처를 추측하지 않고
`legacy_unverified`로 표시하며 정확한 revision으로 재생성해야 verified가 된다.
Linux에서는 같은 테스트 바이너리가 eager와 streaming을 각각 별도 모델 핸들로 실행해
PyTorch 기준 오차와 `execution_mode_error_vs_eager`를 함께 기록한다. 이 비교는 같은
모델·corpus·pooling·normalize·thread policy를 사용하며 성능 timed section 밖의
report-only 정확도 산출물이다. Q4도 계속 report-only다.

### 저장된 M4 측정 결과

[M4 Docker Desktop arm64 closeout](bench/results/M4-docker-desktop-arm64/CLOSEOUT.md)은
M3.6 baseline과 같은 Docker Desktop 4.38.0 Ubuntu 24.04 `linux/arm64` VM/container,
`/src`·`/build` mount에서 수집한 40 result JSON, 40 raw sidecar와 정확도 보고서를
보존한다. 이 범위를 벗어난 물리 target 또는 일반 Linux 성능으로 해석하면 안 된다.

동일 M4 binary의 authoritative warm/profile-off 결과는 다음과 같다.

| workload | eager→streaming lifetime RSS | throughput 변화 |
|---|---:|---:|
| BERT F16 short | 79.07→14.90 MiB (-81.2%) | -3.4% |
| BERT F16 long | 91.94→34.02 MiB (-63.0%) | +0.1% |
| Harrier F32, 세 corpus | 약 1112→102 MiB (약 -90.8%) | -4.5~-17.2% |
| Harrier Q8_0, 세 corpus | 약 361→87 MiB (약 -76.0%) | -21.7~-38.0% |
| Harrier Q4_K | 313.26→84.62 MiB (-73.0%) | -14.0% (report-only) |

strict-cold의 canonical `startup_to_first_result_ms`는 Harrier streaming에서
12.9~48.7% 줄었지만 inference-only latency와 page fault는 대체로 늘었다. 모든 cold
row는 `mincore`로 worker 시작 전 resident page 0을 검증했지만 `posix_fadvise`와
`madvise`는 advisory이며 Docker 아래 host cache까지 증명하지 않는다.

profile-on은 10 ms `smaps_rollup`으로 PSS/USS를 수집한 별도 pass다. 이 peak는 sampled
lower bound이고 sampler가 page table과 residency를 건드릴 수 있어 latency는
diagnostic이다. F32 activation ping-pong과 요청 token row가 사용하는 embedding-table
page는 streaming에서도 필요하다. 결과는 `independent_runs: 1`, null CI이고 golden
provenance는 정확한 upstream revision을 복구하지 못한 `legacy_unverified`다. 기존 gate는
변경하지 않았다.

측정 지표와 메모리 수치의 의미는 [테스트와 벤치마크 가이드](docs/guide/26080401/07-tests-bench-workflow.md)에서 설명한다.
