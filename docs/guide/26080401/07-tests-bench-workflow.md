[← 06. API와 소유권](06-api-and-ownership.md) | [08. 코드 규칙과 확장 →](08-conventions-and-extending.md)

# 07. 테스트와 벤치마크

NanoEmbed의 테스트는 단순히 최종 벡터 하나가 나오는지 확인하지 않는다. 토큰화, 각 BERT 레이어, 전체 공개 API와 메모리 재사용을 서로 다른 경계에서 검사한다.

경계를 나누는 이유는 실패 원인을 좁히기 위해서다. 최종 임베딩만 다르다는 사실로는 토큰 ID가 틀렸는지, 어텐션 계산이 틀렸는지, 풀링 설정이 다른지 알기 어렵다.

## 기본 빌드와 테스트

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

특정 테스트만 실행할 수도 있다.

```sh
ctest --test-dir build -R tokenizer --output-on-failure
ctest --test-dir build -R 'golden|seq_len|limits' --output-on-failure
```

## 모델 파일이 필요한 테스트

저장소에는 GGUF 모델을 커밋하지 않는다. 모델이 없으면 일부 테스트가 환경 변수 부재를 확인하고 해당 검사를 건너뛴다.

기본 모델 경로는 다음과 같다.

```text
models/bge-small-en-v1.5-f16.gguf
```

CMake가 이 파일을 찾으면 각 테스트에 필요한 환경 변수를 자동으로 설정한다.

| 테스트 | 환경 변수 |
|---|---|
| `scanner` | `NANOEMBED_TEST_MODEL` |
| `tokenizer` | 모델 + `NANOEMBED_TOKENIZER_FIXTURE` |
| `forward` | 모델 + `NANOEMBED_ACTIVATION_FIXTURES` |
| `golden` | 모델 + `NANOEMBED_GOLDEN_FIXTURE` |
| `seq_len` | `NANOEMBED_TEST_MODEL` |
| `limits` | `NANOEMBED_TEST_MODEL` |

### 로컬에서 건너뛰기를 허용하는 이유

새로 clone한 개발자가 모델을 받지 않아도 빌드, 순수 C ABI와 합성 GGUF 오류 테스트를 실행할 수 있게 하기 위해서다. 모델 파일은 크고 Git에 포함되지 않는다.

### CI에서 건너뛰기를 허용하지 않는 이유

모델이 필요한 테스트가 전부 건너뛰었는데도 `ctest`가 성공하면 정확도 검증 없이 CI가 초록색이 될 수 있다.

CI는 모델을 내려받아 캐시하고 다음 옵션으로 구성한다.

```sh
-DNANOEMBED_REQUIRE_MODEL=ON
```

이 옵션에서는 모델 파일이 없으면 CMake 구성 단계가 실패한다.

## 테스트 목록과 책임

CTest에는 모델 기능 테스트 외에도 benchmark 통계·파서·Python orchestration 계약
검사가 등록된다. Linux에서는 `/proc`을 실제로 읽는 selftest가 실행되고,
비-Linux에서는 같은 이름의 테스트가 `Skipped`로 남아 live 검증을 했다고 오해하지
않게 한다.

| CTest 이름 | 파일 | 확인하는 경계 |
|---|---|---|
| `abi_link` | `tests/abi/abi_link_test.c` | 공개 C 헤더, 심볼과 기본 오류 동작 |
| `scanner` | `tests/unit/scanner_test.cpp` | GGUF 메타데이터와 BERT 텐서 검사 |
| `tokenizer` | `tests/unit/tokenizer_test.cpp` | 문자열에서 토큰 ID까지 |
| `forward` | `tests/unit/forward_test.cpp` | 임베딩 레이어와 각 BERT 블록 출력 |
| `golden` | `tests/integration/golden_test.cpp` | 공개 API 전체 경로의 최종 임베딩 |
| `golden_accuracy` | `tests/unit/golden_accuracy_test.cpp` | 오차 통계와 fixture provenance 검증 |
| `seq_len` | `tests/integration/seq_len_test.cpp` | 긴 입력과 그래프 버퍼 재사용 |
| `limits` | `tests/integration/limits_test.cpp` | 입력 상한과 미지원 설정 거부 |
| `mode_selection` | `tests/integration/mode_selection_test.cpp` | strict eager/streaming 선택, 최초 context mode lock, 실패 복구와 동시 context |
| `bench_statistics` / `bench_metrics` | benchmark unit tests | percentile·분산과 `/proc` 텍스트 파싱 |
| `bench_runner` / `bench_harness_integration` | Python unit/integration | corpus·process topology·profile·schema·sidecar 조합 |
| `bench_selftest` | `nanoembed-bench --selftest` | Linux profile-off, fork+exec, RSS/VmHWM 관찰 |
| `bench_selftest_memory_profile` | `nanoembed-bench --selftest --memory-profile` | Linux `smaps_rollup` PSS/USS sampling |

## `abi_link`: 공개 API의 최소 사용자

이 테스트는 C++이 아니라 C11로 컴파일한다. 공개 헤더가 C 문법으로 유효한지, 모든 공개 심볼이 링크되는지 확인한다.

다음 실패 경로도 검사한다.

- 존재하지 않는 모델 파일
- `NULL` 모델과 컨텍스트
- `NULL`을 허용하는 해제 함수
- 기본 컨텍스트 설정값

공개 헤더나 CMake 링크 구성을 바꿨다면 가장 먼저 볼 테스트다.

## `scanner`: 실제 모델과 합성 GGUF

정상 모델이 있으면 bge-small의 레이어 수, hidden 차원, 헤드 수, FFN 차원, 어휘 크기와 최대 길이를 확인한다.

실패 경로에는 테스트가 직접 만든 작은 GGUF를 사용한다. 큰 모델 파일을 손상시키지 않고도 다음 상황을 정확히 만들 수 있다.

- 잘못된 `general.architecture`
- 필수 메타데이터 누락
- 필수 텐서 누락
- 예상과 다른 텐서 차원 크기

합성 입력은 어떤 오류를 만들었는지 테스트 코드에 그대로 남긴다. 실패하면 재현한 조건을 코드에서 바로 확인할 수 있다.

## `tokenizer`: 토큰 ID 비교

[`bge-small-eval.tsv`](../../../tests/fixtures/tokenizer/bge-small-eval.tsv)는 Hugging Face 토크나이저의 결과를 저장한 테스트 기준 데이터다.

각 행에는 원문과 쉼표로 구분된 토큰 ID가 들어 있다.

```text
hello world<TAB>101,...,102
```

NanoEmbed의 `WordPieceTokenizer::encode()` 결과를 원소별로 비교한다. 특수 토큰 ID와 어휘 크기도 GGUF 메타데이터에서 확인한다.

토큰 ID 하나가 다르면 모든 뒤쪽 활성값이 달라진다. 최종 기준값 테스트보다 먼저 토큰화 경계를 별도로 검사하는 이유다.

## `forward`: 중간 활성값 비교

활성값은 모델 계산 중간에 만들어지는 텐서 데이터다. `forward` 테스트는 Hugging Face BERT에서 저장한 다음 출력을 사용한다.

- 입력 토큰 ID와 어텐션 마스크
- 입력 임베딩 단계 출력
- 0번부터 11번까지 각 인코더 블록 출력

파일은 `NEMB`라는 작은 바이너리 형식이다.

```text
"NEMB"
버전
텐서 개수
각 텐서:
  이름
  자료형
  차원 수와 각 축의 크기
  원시 데이터
```

Hugging Face 배열은 `[B, S, H]`, ggml 활성값은 `[H, S, B]`로 표현한다. 테스트는 같은 메모리의 축 의미가 대응되도록 텐서를 만든다.

각 BERT 블록을 검사할 때 NanoEmbed의 이전 블록 출력을 다음 입력으로 사용하지 않는다. Hugging Face의 이전 블록 출력을 직접 넣는다.

```text
HF layer 4 출력
  → NanoEmbed layer 5만 실행
  → HF layer 5 출력과 비교
```

앞 레이어의 작은 오차가 계속 누적되는 것을 막고, 어느 블록 자체가 처음 잘못되었는지 찾기 위한 방식이다.

현재 최대 절대 오차 허용값은 `5e-3`이다. F16 가중치와 여러 값을 더하는 순서의 차이를 고려한 값이며 테스트 코드의 상수가 최종 기준이다.

## `golden`: 최종 임베딩 기준값 테스트

실제 파일명은 역사적으로 `golden_test.cpp`다. 문서에서는 역할이 드러나도록 기준값 테스트라고 부른다.

`NEGD` 파일에는 문장과 sentence-transformers가 만든 최종 임베딩이 함께 들어 있다.

```text
"NEGD"
버전
문장 개수
임베딩 차원
각 문장:
  UTF-8 문자열
  float32 기준 임베딩
```

테스트는 공개 C API로 같은 문장을 실행하고 코사인 유사도를 계산한다.

| 기준 | 요구값 |
|---|---:|
| 문장별 최소 코사인 유사도 | 0.9999 |
| 전체 평균 코사인 유사도 | 0.99999 |

sentence-transformers는 bge-small에 CLS 풀링을 사용한다. 공개 API 기본값은 Mean이므로 테스트는 `NANOEMBED_POOL_CLS`를 명시한다. 풀링 설정이 다르면 인코더 계산이 맞아도 최종 벡터가 다르다.

현재 CTest는 `NANOEMBED_ACCURACY_JSON`을 설정해 `build/accuracy-report.json`도
만든다. 이 파일은 성능 benchmark timed section과 분리된 정확도 산출물이다.

- PyTorch F32 기준과의 cosine similarity, maximum/mean absolute error, RMSE,
  output/reference norm과 절대·상대 norm 차이를 문장별로 기록한다.
- 각 지표의 worst와 p50/p90/p95/p99 집계를 기록한다. 새 지표는 report-only이며
  기존 cosine gate를 바꾸지 않는다.
- F32 구현 오차(`pytorch_reference_error`)와 Q8/Q4가 native F32에서 추가로
  벌어진 양자화 손실(`quantization_loss_vs_native_f32`)을 섞지 않는다.
- Linux에서는 F16/F32/Q8/Q4 각각을 streaming public API로 다시 실행한다. 동일
  binary·모델·fixture·pooling·normalize·thread policy의 eager 출력과 비교한 full-vector
  지표는 `execution_mode_error_vs_eager`에 별도로 기록한다.

각 `.bin` 옆의 `.provenance.json`과 `.provenance.sha256`은 fixture와 manifest
hash, model ID와 정확한 Hugging Face commit, CPU FP32·pooling·normalize·truncation,
seed·batch size·thread 수, generator 명령과 패키지 lock을 묶는다. `golden` 테스트는
사용 전에 이 무결성을 검사한다. 과거 fixture처럼 정확한 revision을 복구할 수 없는
자료는 추정값으로 채우지 않고 `legacy_unverified`로 명시한다.

## `seq_len`: 그래프 버퍼 재사용

초기 M3 구현은 길이 약 270을 넘는 입력에서 고정 256 MiB 메모리 영역이 부족해 프로세스가 종료될 수 있었다. `seq_len` 테스트는 이 문제가 다시 생기지 않는지 확인한다.

- 길이 512에 가까운 입력이 성공한다.
- 512를 넘는 입력은 안전하게 잘린다.
- 긴 입력과 짧은 입력을 번갈아 실행한다.
- 같은 짧은 문장의 실행 전후 결과가 코사인 유사도 0.99999 이상으로 유지된다.

마지막 검사는 재사용 버퍼에 남은 이전 활성값이 다음 그래프를 오염시키지 않는지 확인한다.

## `limits`와 `mode_selection`: API 상한과 실행 모드 계약

`limits` 테스트는 설정과 모델 상한의 관계를 확인한다.

- 컨텍스트 상한에서 실행한 결과가 기대한 잘림 결과와 같다.
- `max_seq_len=100000`을 요청해도 BERT 위치 임베딩 상한 512를 넘지 않는다.
- `use_streaming`은 정확히 `0`(eager) 또는 `1`(Linux streaming)만 허용한다.
- 최초로 성공한 context가 모델 핸들의 모드를 고정하고 same-mode context만 허용한다.
- 실패한 최초 context 생성은 mode lock이나 부분 runner를 남기지 않는다.
- 비-Linux의 `use_streaming=1`과 mixed-mode context는 명시 오류다.
- 서로 다른 streaming context 두 개는 하나의 mapped model을 공유하면서 동시에 실행된다.

지원하지 않는 옵션을 무시하면 호출자는 스트리밍을 사용한다고 생각하면서 실제로는 더 많은 메모리를 쓸 수 있다. 따라서 명시적으로 거부한다.

모델 핸들 생성은 metadata-only descriptor 단계다. `n_embed`, `n_layer`, model context
length와 default pooling query는 mode 선택 전에도 동작한다. 전체 eager 가중치 로드나
streaming mmap/range 분류는 최초 context 생성에서 일어난다. eager와 streaming을 같은
프로세스에서 A/B 비교할 때는 같은 GGUF 경로로 모델 핸들을 두 개 만든다.

## 실패 조합으로 원인 좁히기

| 관찰 | 먼저 확인할 경계 |
|---|---|
| `scanner` 실패 | GGUF 메타데이터, 텐서 이름과 차원 크기 |
| `tokenizer` 실패 | 문자열 정규화, WordPiece 분해, 특수 토큰 |
| 특정 `forward` 블록만 실패 | 해당 레이어의 가중치 연결과 그래프 연산 |
| `forward`는 통과하고 `golden`만 실패 | 토큰화, 풀링, L2 정규화, 전체 조립 |
| 짧은 입력은 통과하고 `seq_len`만 실패 | 그래프 예약, 입력 길이별 버퍼 배치 |
| `limits`만 실패 | 설정 검증과 모델 최대 길이 제한 |

## 테스트 기준 데이터 만들기

일상적인 빌드에서는 커밋된 기준 데이터를 다시 만들 필요가 없다. 모델, 코퍼스, 토크나이저 동작이나 비교 대상 프레임워크를 바꿀 때 재생성한다.

개발용 Python 환경을 먼저 만든다.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

| 스크립트 | 생성물 | 사용하는 테스트 |
|---|---|---|
| `tools/dump_corpus.py` | `tests/corpus/eval.txt` | 기준값 생성과 벤치 입력 |
| `tools/dump_tokenizer_fixture.py` | 토큰 ID TSV | `tokenizer` |
| `tools/dump_hf_activations.py` | `*.nemb`와 원문 `*.txt` | `forward` |
| `tools/dump_golden.py` | 최종 임베딩 `.bin` + provenance manifest/checksum | `golden` |

코퍼스를 바꾸면 같은 문장을 사용하는 토큰 ID와 최종 임베딩 기준 데이터도 다시 만들어야 한다. 모델을 바꾸면 토큰화, 활성값과 최종 임베딩 기준 데이터를 모두 새 모델로 생성해야 한다.

커밋할 golden fixture는 lower-bound 개발 환경이 아니라 정확히 고정된 lock으로
생성한다. revision은 필수이며 generator가 실제 commit SHA로 해석해 manifest에
기록한다.

```sh
python3 -m venv .venv-golden
.venv-golden/bin/pip install -r requirements-golden.lock
.venv-golden/bin/python tools/dump_golden.py \
  --model BAAI/bge-small-en-v1.5 \
  --revision <exact-Hugging-Face-commit> \
  --corpus tests/corpus/eval.txt \
  --out tests/fixtures/golden/st-bge-small.bin \
  --pooling cls \
  --normalize \
  --batch-size 32 \
  --max-length 512 \
  --seed 0 \
  --torch-threads 1
```

캐시만 사용하려면 `--local-files-only`를 추가한다. exact revision이나 lock이 맞지
않으면 생성기는 실패하며, 기존 `legacy_unverified` manifest를 verified로 바꾸지
않는다.

## 벤치마크가 측정하는 것

기능 테스트는 결과가 맞는지 확인하고 benchmark는 latency, throughput, RSS/PSS/USS,
page fault와 I/O를 측정한다. 벤치마크 도구는 Linux의 `/proc` 정보에 의존한다.
macOS에는 같은 의미의 `VmHWM` reset과 `smaps_rollup`이 없으므로 native tool은
Linux에서만 빌드한다. portable unit/runner integration은 macOS에서도 실행하지만
Linux selftest는 명시적으로 skip된다.

### 왜 별도 작업 프로세스를 사용하는가

측정 도구가 모델을 직접 로드하면 측정 준비에 사용한 메모리까지 결과에 섞일 수 있다.

부모 프로세스는 모델을 로드하지 않는다. 자신을 `fork`한 뒤 `exec`해 깨끗한 주소 공간의 작업 프로세스를 만들고, 부모가 바깥에서 작업 프로세스의 메모리를 읽는다.

```text
부모 측정 프로세스
  ├─ fork + exec
  ├─ 작업 프로세스 /proc 관찰
  └─ 결과 JSON 구성

warm 작업 프로세스
  ├─ 모델 로드와 context 생성
  ├─ warmup
  ├─ GO 이후 측정 반복
  └─ CPU·page fault·지연 시간 보고

cold 선택 입력 하나
  └─ 새 nanoembed-bench → 새 fork+exec worker → 첫 inference 1회
```

단순 `fork`는 부모가 사용한 페이지를 copy-on-write 상태로 자식에게 보인다.
`exec`은 주소 공간을 새 프로그램으로 교체한다. warm은 한 worker에서 warmup 후 반복하고,
cold는 선택된 입력마다 Python runner가 native 프로세스를 다시 실행한다. native cold
worker는 GO 이후 model load → context 생성 → 첫 inference를 수행한다.

M4 lazy loader에서는 `model_load_ms`가 metadata descriptor 생성만 재고,
`context_create_ms`가 선택 모드의 실제 초기화(eager 전체 가중치 로드 또는 streaming
mmap·검증·range 분류)를 포함한다. 따라서 두 phase를 단독으로 과거 eager loader와
비교하면 lifecycle 이동을 성능 변화로 오해한다. cold의 canonical
`startup_to_first_result_ms`는 두 phase와 첫 inference를 함께 포함한다.

warm 결과의 latency population에는 `warmup_items_executed`가 포함되지 않는다. cold
결과의 `cold_phase_timings`는 model load, context 생성, 첫 request, startup-to-first-result를
나누고 `cold_worker_invocations`와 입력별 audit trail을 보존한다. cold throughput의
분모는 각 worker의 GO-to-first-result 합이며 cache eviction과 process launch는 제외한다.

### cold cache를 검증하는 방법

부모는 worker를 만들기 전에 model file에 `posix_fadvise(DONTNEED)`를 요청하고
`mincore`로 resident page를 센다. `cold_cache_verified`는 eviction 뒤, worker fork 전
resident page가 0일 때만 true다. 기본 모드는 실패를 `cold-unverified`로 기록해 수치를
남기고, `--strict-cold`는 검증되지 않으면 worker 시작 전에 실패한다. worker 실행 뒤
residency는 설명용이며 시작 상태를 소급해 바꾸지 않는다.

### 두 종류의 최대 RSS

- `rss_peak_lifetime_mb`: 모델 로드와 warmup을 포함한 작업 프로세스 전체 생애의 최대 RSS
- `rss_peak_window_mb`: 실제 반복 측정 구간 안의 최대 RSS

Linux의 `VmHWM`은 한 번 커지면 스스로 내려가지 않는다. 측정 구간 직전에 `/proc/<pid>/clear_refs`에 `5`를 써 현재 RSS로 high-water mark를 재설정한다. 재설정 전 값도 읽어 두 종류를 모두 보존한다.

### RSS, PSS와 USS

| 값 | 답하려는 질문 |
|---|---|
| RSS | 이 프로세스를 실행하려면 실제 RAM이 대략 얼마나 필요한가? |
| PSS | 공유 페이지 비용을 나눠 계산하면 프로세스 몫은 얼마인가? |
| USS | 이 프로세스를 종료했을 때 단독으로 회수되는 메모리는 얼마인가? |

M3에서는 가중치를 일반 메모리에 올리므로 세 값이 비슷하다. M4에서 GGUF를 `mmap`하면 파일 페이지가 여러 프로세스 사이에 공유될 수 있어 차이가 커진다.

상세 메모리 profile은 기본 off다.

| 실행 | timed path 관찰 | 해석 |
|---|---|---|
| profile-off | 경계의 값싼 `statm` RSS + 커널 `VmHWM`; sampler thread와 `smaps_rollup` 없음 | latency/throughput authoritative |
| profile-on | `--memory-profile-interval-ms` 간격의 `smaps_rollup` + final sample | PSS/USS 최종 보고용, latency diagnostic |

PSS와 USS에는 kernel high-water mark가 없다. 따라서 `pss_peak_sampled_mb`와
`uss_peak_sampled_mb`는 관측된 표본의 최대이며 샘플 사이의 짧은 피크를 놓칠 수 있는
lower bound다. 간격을 줄이면 놓칠 가능성은 낮아지지만 `smaps_rollup`이 page table을
순회하고 target의 memory lifecycle/page residency를 교란할 가능성은 커진다. 결과의
attempted/effective sample 수, valid ratio, read-duration과 observer-effect 필드를 함께
검토한다. profile-on latency를 성능 baseline으로 사용하지 않는다.

profile-on main result의 `rss|pss|uss_sampled_p50|p75|p90|p95|p99_mb`와
`rss|pss|uss_avg_mb`는
valid periodic 표본과 valid mandatory final 표본을 합친 population에서 계산한다. baseline은
포함하지 않는다. native result의 mean/sampled peak도 같은 population을 사용한다. cold
runner의 sampled mean과 percentile은 모든 worker 원본 population을 정확히 합쳐 계산하며,
baseline/final 경계값은 worker별 평균, peak는 worker별 peak의 최대를 유지한다. percentile은 아래
latency 통계와 같은 lower 규약이다. profile-off에서는 이 필드가 모두 null이고 sampler도
시작하지 않는다.

### latency와 throughput 통계

요청별 latency는 min/max/mean, p50/p90/p95/p99, population stddev와 MAD를 기록한다.
percentile은 정렬한 표본에서 `floor(q * (count - 1))`인 lower 방식을 사용한다.
schema v3에서 `items_per_sec`와 `batches_per_sec`는 canonical timed wall을 기준으로 한
전체 처리율이다. `batch_latency_*`는 API batch 한 번, `item_latency_*`는 해당 시간을
그 batch의 item 수로 나눈 환산값이다. `single_request_items_per_sec`는 batch size 1의
호환 필드로만 남는다.
고정 10-item window throughput 통계는 표본 window가 충분할 때만 제공하며 canonical이
아니다. 결과의 `collection_status`와 null을 0으로 해석하지 않는다.

### eager/streaming A/B의 실행 모드 증거

scenario의 `streaming: false|true`는 native `--streaming`과 frozen C ABI의
`nanoembed_context_params.use_streaming`으로 이어진다. 결과에는 flat field와
`settings`, `execution_mode_resolution`에 requested/resolved mode가 중복 기록된다.
worker는 strict context 생성에 성공했을 때만 resolved mode를 선언하고, parent와 Python
runner는 모든 claim이 일치하지 않으면 실행을 실패시킨다. public ABI에 이를 위한 새
diagnostic 함수는 추가하지 않는다.

`compare.py`는 execution-mode 차이를 comparability diagnostic으로 표시한다. M4의
eager/streaming 차이를 해석할 때는 의도한 축인지 확인하고, 그 밖의 model·corpus·pooling·
normalize·threads·binary/build fingerprint는 같아야 한다. profile-off 결과가 성능용이고
profile-on 결과는 PSS/USS 보고용이라는 구분은 두 모드에서 동일하다.

## 시나리오 실행

Linux Release 빌드와 Python 개발 의존성이 필요하다.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

profile-off warm 실행은 성능 해석용이다. 그룹 전체가 너무 크면 `NAME:N`으로
제한한다.

```sh
.venv/bin/python bench/runner.py \
  --milestone M3.6 \
  --group english_short:20 \
  --selection-seed 0 \
  --cache-state warm \
  --raw-samples-out bench/results/local.samples.json \
  --out bench/results/local-perf.json
```

여러 workload를 한 번에 선택하고 공통 N을 적용할 수 있다. 명시적인 `NAME:N`이
`--samples-per-group`보다 우선한다. 어느 N도 없으면 각 group 전체를 실행한다.

```sh
.venv/bin/python bench/runner.py \
  --milestone M3.6 \
  --filter single_short_f16 \
  --group english_short \
  --group unicode_edge:5 \
  --samples-per-group 20 \
  --selection-seed 0 \
  --out bench/results/groups.json
```

상세 메모리와 strict cold는 별도 pass로 수집한다.

```sh
.venv/bin/python bench/runner.py \
  --milestone M3.6 \
  --filter single_short_f16 \
  --group english_short:20 \
  --cache-state warm \
  --memory-profile \
  --memory-profile-interval-ms 10 \
  --out bench/results/memory.json

.venv/bin/python bench/runner.py \
  --milestone M3.6 \
  --filter single_short_f16 \
  --group english_short:5 \
  --cache-state cold \
  --strict-cold \
  --out bench/results/cold.json
```

`runner.py`는 `bench/scenarios.yaml`에서 milestone을 포함한 정의만 선택한다. corpus는
`bench/corpus_groups.json`의 유형별 source로 구성되며 선택된 text ID, source,
selection seed와 SHA-256이 결과에 남는다. 같은 seed와 corpus bytes는 같은 subset을
고른다. `--raw-samples-out` sidecar에는 요청별 latency와 profile-on의 원본 memory
timeline을 저장하고 main JSON에는 sidecar 경로·크기·hash만 기록한다. memory 표본은
bytes 단위 RSS/PSS/USS와 smaps breakdown, `sample_role`, monotonic timestamp,
rollup read duration을 갖는다. GO를 `elapsed_ms_from_go=0`으로 삼으므로 baseline은
음수이고 periodic/final은 0 이상이며 DONE marker 시각도 보존된다. cold 결과는 worker마다
별도의 process-local monotonic epoch와 GO 원점을 유지하므로 worker 사이 timestamp를
한 시간축처럼 정렬하면 안 된다. profile-off sidecar의 memory collection은 `disabled`,
`samples: null`이며 성능 실행에서 sidecar를 요청하지 않으면 native raw 직렬화도 하지
않는다. native raw schema 1에서 `memory_profile`은 하위호환 optional extension이어서
이 필드가 없는 옛 sidecar는 `unavailable_legacy`로 읽는다.

### 결과 contract와 한계

schema v3는 run 수준 requested/resolved settings, batch size/max_batch, UTC,
full Git/ggml SHA와 dirty 상태,
benchmark/model/manifest/scenario hash, build/compiler/CMake option, CPU governor·NUMA·RAM·
filesystem fingerprint를 저장한다. scenario별 model과 selected-input identity도 있다.
필수 artifact hash 실패는 native 실행 전에 실패하고, 지원되지 않는 optional host
정보는 `unavailable`/null이다.

`aggregates.by_group`와 `overall`은 기술 통계다. 서로 다른 model/cache/profile을 섞으면
`homogeneous_dimensions`가 false이고 개별 scenario가 canonical이다. raw sample 없이
scenario percentile을 합쳐 새 percentile을 만들 수 없으므로 aggregate percentile은
null이다. 한 runner 결과는 `independent_runs: 1`, `confidence_interval: null`이다.
같은 process 안의 많은 request가 여러 독립 run을 대신하지 않는다.

## 기준 측정값과 비교

```sh
.venv/bin/python bench/compare.py \
  bench/baseline/M3.5.json \
  bench/results/local-perf.json \
  --strict
```

비교기는 unversioned M3/M3.5 결과를 schema v1로 읽고 v2의
batch size 1에서는 `single_request_items_per_sec`와 옛 `throughput_items_per_sec`
alias를 연결한다. 먼저
다음 핵심 환경을 확인한다.

- Linux 커널
- CPU 모델
- 논리 CPU 수
- 페이지 크기

환경이 다르면 같은 코드라도 지연 시간과 메모리가 달라질 수 있다. 기본 모드는 경고 후 표를 출력하고, `--strict`는 환경 불일치에서 종료 코드 2를 반환한다.

Git/model/input/build/CPU policy 등 확장 fingerprint 차이는 별도 진단으로 출력하지만
기존 gate를 넓히지 않는다. 동일하지 않은 fingerprint를 무시하고 수치를 M4 개선으로
해석해서는 안 된다.

현재 비교 정책은 다음과 같다.

- p50, p90, p99 지연 시간의 15% 초과 악화는 회귀로 보고한다.
- 15% 초과 개선도 측정 조건 변화인지 검토할 수 있도록 별도 개선 항목으로 보고한다.
- 전체 생애 최대 RSS의 5% 초과 증가는 회귀로 보고한다.
- PSS, USS와 page fault는 표에 출력하지만 자동 실패 조건으로 사용하지 않는다.
- 기준에 있던 시나리오가 현재 결과에서 사라지면 불완전한 실행으로 보고한다.
- `--strict`에서만 회귀가 비정상 종료 코드 1을 만든다.

지연 시간이 크게 좋아졌다고 기준값을 즉시 바꾸면 안 된다. 입력 개수, 스레드 수, 풀링, 모델과 측정 방식이 같았는지 먼저 확인한다.

## M4 저장 결과와 해석

최종 M4 bundle은
[`bench/results/M4-docker-desktop-arm64`](../../../bench/results/M4-docker-desktop-arm64/ENVIRONMENT.md)에
있다. M3.6 baseline을 만든 동일 Docker Desktop 4.38.0 Ubuntu 24.04 arm64
VM/container와 `/src`·`/build` bind mount에서 한 번의 고정 Release binary로 수집했다.
물리 Linux target이나 다른 Docker VM으로 일반화하는 기준값은 아니다.

저장 matrix는 eager/streaming 각각에 대해 다음 네 pass를 분리한다.

1. warm/profile-off: authoritative latency와 throughput
2. cold/profile-off + `--strict-cold`: authoritative first-request/startup
3. warm/profile-on 10 ms: PSS/USS memory report, latency diagnostic
4. cold/profile-on 10 ms + `--strict-cold`: cold PSS/USS memory report, latency diagnostic

BERT F16 short/long과 Harrier F32/Q8_0의 `english_short`, `multilingual_short`,
`vocab_spread`는 M3.6과 같은 selection hash/N을 사용한다. warm/profile-off BERT short만
N=3이고 나머지는 N=1이다. Q4_K는 pre-M4 baseline이 없는 M4 내부 report-only 축이다.
40 bounded result JSON과 40 raw sidecar에 총 72 workload row가 있고, 모든 36 cold row가
worker 시작 전 guest-visible model page 0을 확인했다.

### 같은 M4 binary의 streaming 효과

warm/profile-off의 lifetime RSS와 `single_request_items_per_sec` 변화는 다음과 같다.

| workload | eager RSS → streaming RSS | throughput 변화 |
|---|---:|---:|
| BERT F16 short | 79.07 → 14.90 MiB (-81.2%) | -3.4% |
| BERT F16 long | 91.94 → 34.02 MiB (-63.0%) | +0.1% |
| Harrier F32, 3 groups | 1111.97~1113.90 → 102.24~102.94 MiB (약 -90.8%) | -4.5~-17.2% |
| Harrier Q8_0, 3 groups | 360.83~363.04 → 87.04~87.05 MiB (약 -76.0%) | -21.7~-38.0% |
| Harrier Q4_K | 313.26 → 84.62 MiB (-73.0%) | -14.0%, report-only |

profile-on warm sampled peak에서도 Harrier F32 PSS는 약 1110~1116 MiB에서
100~101 MiB, USS는 약 1109~1115 MiB에서 99 MiB 안팎으로 줄었다. Harrier Q8_0은
PSS가 약 359~361 MiB에서 85 MiB 안팎, USS가 약 358~360 MiB에서 84 MiB 안팎으로
줄었다. 이 값은 10 ms 표본 최대라 실제 순간 peak의 lower bound다.

cold/profile-off에서는 canonical `startup_to_first_result_ms`가 Harrier에서
12.9~48.7%, BERT short에서 7.0%, BERT long에서 1.7% 줄었다. 반면 streaming은 각
phase가 mapped weight page를 처음 만질 때 fault를 내므로 inference-only latency와
major/minor page fault가 대체로 증가했다. Docker가 보고한 `io_read_bytes`는 전 row에서
0이었지만 lower host cache/storage read가 없었다는 뜻은 아니다.

### 두 비교 층을 섞지 않는다

[`comparison-summary.json`](../../../bench/results/M4-docker-desktop-arm64/comparison-summary.json)은
두 종류의 pair를 따로 저장한다.

- M4 eager 대 M3.6 32 rows: harness/lifecycle/code drift 진단
- 같은 M4 binary의 streaming 대 eager 36 pairs: 기능 효과

M4 eager warm RSS는 M3.6에서 약 1.1% 이내였지만 Q8_0 warm throughput은 group에 따라
+2.7%에서 -20.4%까지 달랐다. cold startup drift도 +0.05~+62.8%였다. metadata-only
model handle로 인해 eager weight 준비가 `model_load_ms`에서 `context_create_ms`로
이동했으므로 두 phase를 따로 회귀로 해석하지 않는다. cold의 canonical 값은 전체
`startup_to_first_result_ms`다. 한 independent run만으로 남은 drift의 통계적 유의성이나
원인을 확정하지 않는다.

### 정확도와 남은 메모리 바닥

accuracy CTest는 performance run과 분리해 F16/F32/Q8/Q4 eager·streaming 8개 비교를
저장했다. streaming 대 eager의 full-vector maximum/mean absolute error, RMSE와 norm
difference는 전부 0이고 worst cosine은 부동소수점 반올림 범위에서 1이다. streaming 대
PyTorch worst cosine은 F16 0.999999156, F32 0.999999725, Q8_0 0.999105757,
report-only Q4_K 0.947608256이며 기존 cosine gate는 그대로 통과했다.

M4는 transformer layer weight page를 줄이지만 두 메모리 원천은 남는다.

- 레이어 사이 활성값은 context-local F32 ping-pong buffer 두 개에 보관한다.
- token embedding table은 layer가 아니다. 전체 table을 anonymous buffer로 복사하지는
  않지만 현재 요청 token ID의 row page는 실제로 읽어야 한다.

`madvise`와 `posix_fadvise`는 advisory이고 `mincore`는 Docker guest에서 관찰한 mapping
residency만 설명한다. PSS/USS는 sampled lower bound이고 profile-on sampler의
`smaps_rollup` page-table walk는 observer effect가 있다. bundle은
`independent_runs: 1`, `confidence_interval: null`이며 통계적 유의성을 주장하지 않는다.
golden fixture의 정확한 Hugging Face revision과 원래 생성 환경은 복구하지 못해
provenance가 `legacy_unverified`다. Q4와 새 memory/error metric은 report-only이며 기존
gate를 바꾸지 않았다. 전체 수치와 checksum/검증 로그는
[`CLOSEOUT.md`](../../../bench/results/M4-docker-desktop-arm64/CLOSEOUT.md)에 있다.

## `bench_selftest`

측정 도구 자체가 잘못되면 모델 수치도 믿을 수 없다. Linux CTest의 두 selftest는
GGUF 없이 합성 메모리 할당을 사용한다.

- 부모가 64 MiB를 먼저 사용해도 작업 프로세스 시작 RSS에 섞이지 않는지 확인한다.
- `clear_refs`로 측정 구간 최대 RSS를 재설정할 수 있는지 확인한다.
- profile-off에서 작업 프로세스가 할당한 64 MiB만큼 RSS가 증가하고, sampler attempt가
  0이며 baseline/final에 PSS가 없어 timed path가 `smaps_rollup`을 읽지 않았음을 확인한다.
- profile-on에서 여러 valid `smaps_rollup`과 final sample을 얻고 USS 증가를 포착하며
  JSON latency role이 diagnostic인지 확인한다.

`bench_harness_integration`은 모든 OS에서 실제 runner CLI와 단기 executable fixture를
사용해 cold input별 별도 native PID/worker count, warmup 제외, profile on/off, 단일/복수
group과 N override, raw sidecar, schema v1→v2 비교를 검사한다. 이 테스트와 selftest는
모델 성능을 재는 테스트가 아니다. Linux가 아니면 native selftest는 성공으로 가장하지
않고 CTest에 `Skipped`로 표시된다.

## 이 장의 핵심 정리

- 토큰, 레이어와 최종 출력 기준을 나눠야 실패 위치를 좁힐 수 있다.
- 모델이 없으면 로컬 일부 검사는 건너뛰지만 CI는 모델을 필수로 요구한다.
- `forward` 테스트는 각 블록에 Hugging Face 입력을 직접 넣어 오차 누적을 분리한다.
- `seq_len`은 큰 버퍼 재사용과 장문 안전성을 확인한다.
- 벤치마크는 `fork+exec` 작업 프로세스를 바깥에서 관찰한다.
- warm과 cold의 process topology와 latency scope가 다르며 cold는 residency를 검증한다.
- profile-off 성능 수치와 profile-on PSS/USS 보고를 분리하고 sampled peak를 lower bound로 해석한다.
- 성능 결과는 같은 환경에서 얻은 기준 측정값과 비교해야 한다.
- 한 결과 파일은 독립 run 하나이므로 confidence interval을 제공하지 않는다.
- RSS, PSS와 USS는 서로 다른 메모리 질문에 답한다.

[← 06. API와 소유권](06-api-and-ownership.md) | [08. 코드 규칙과 확장 →](08-conventions-and-extending.md)
