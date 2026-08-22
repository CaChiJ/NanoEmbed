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

기본 CTest는 현재 7개다. Linux에서는 벤치마크 도구 자체를 검사하는 `bench_selftest`가 하나 더 등록된다.

| CTest 이름 | 파일 | 확인하는 경계 |
|---|---|---|
| `abi_link` | `tests/abi/abi_link_test.c` | 공개 C 헤더, 심볼과 기본 오류 동작 |
| `scanner` | `tests/unit/scanner_test.cpp` | GGUF 메타데이터와 BERT 텐서 검사 |
| `tokenizer` | `tests/unit/tokenizer_test.cpp` | 문자열에서 토큰 ID까지 |
| `forward` | `tests/unit/forward_test.cpp` | 임베딩 레이어와 각 BERT 블록 출력 |
| `golden` | `tests/integration/golden_test.cpp` | 공개 API 전체 경로의 최종 임베딩 |
| `seq_len` | `tests/integration/seq_len_test.cpp` | 긴 입력과 그래프 버퍼 재사용 |
| `limits` | `tests/integration/limits_test.cpp` | 입력 상한과 미지원 설정 거부 |
| `bench_selftest` | `nanoembed-bench --selftest` | Linux 측정 프로세스와 RSS 관찰 방식 |

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

## `seq_len`: 그래프 버퍼 재사용

초기 M3 구현은 길이 약 270을 넘는 입력에서 고정 256 MiB 메모리 영역이 부족해 프로세스가 종료될 수 있었다. `seq_len` 테스트는 이 문제가 다시 생기지 않는지 확인한다.

- 길이 512에 가까운 입력이 성공한다.
- 512를 넘는 입력은 안전하게 잘린다.
- 긴 입력과 짧은 입력을 번갈아 실행한다.
- 같은 짧은 문장의 실행 전후 결과가 코사인 유사도 0.99999 이상으로 유지된다.

마지막 검사는 재사용 버퍼에 남은 이전 활성값이 다음 그래프를 오염시키지 않는지 확인한다.

## `limits`: API 길이 상한과 미지원 기능

`limits` 테스트는 설정과 모델 상한의 관계를 확인한다.

- 컨텍스트 상한에서 실행한 결과가 기대한 잘림 결과와 같다.
- `max_seq_len=100000`을 요청해도 BERT 위치 임베딩 상한 512를 넘지 않는다.
- `use_streaming=1`은 조용히 인메모리 경로로 실행되지 않고 컨텍스트 생성에서 실패한다.

지원하지 않는 옵션을 무시하면 호출자는 스트리밍을 사용한다고 생각하면서 실제로는 더 많은 메모리를 쓸 수 있다. 따라서 명시적으로 거부한다.

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
| `tools/dump_golden.py` | 최종 임베딩 `.bin` | `golden` |

코퍼스를 바꾸면 같은 문장을 사용하는 토큰 ID와 최종 임베딩 기준 데이터도 다시 만들어야 한다. 모델을 바꾸면 토큰화, 활성값과 최종 임베딩 기준 데이터를 모두 새 모델로 생성해야 한다.

## 벤치마크가 측정하는 것

기능 테스트는 결과가 맞는지 확인한다. 벤치마크는 메모리와 실행 비용이 어떻게 변했는지 측정한다.

벤치마크 도구는 Linux의 `/proc` 정보에 의존한다. macOS에는 같은 의미로 최대 RSS를 재설정하는 인터페이스가 없으므로 도구 자체를 Linux에서만 빌드한다.

### 왜 별도 작업 프로세스를 사용하는가

측정 도구가 모델을 직접 로드하면 측정 준비에 사용한 메모리까지 결과에 섞일 수 있다.

부모 프로세스는 모델을 로드하지 않는다. 자신을 `fork`한 뒤 `exec`해 깨끗한 주소 공간의 작업 프로세스를 만들고, 부모가 바깥에서 작업 프로세스의 메모리를 읽는다.

```text
부모 측정 프로세스
  ├─ fork + exec
  ├─ 작업 프로세스 /proc 관찰
  └─ 결과 JSON 구성

작업 프로세스
  ├─ 모델 로드와 준비
  ├─ warmup
  ├─ 측정 반복
  └─ CPU·page fault·지연 시간 보고
```

단순 `fork`는 부모가 사용한 페이지를 copy-on-write 상태로 자식에게 보인다. `exec`은 주소 공간을 새 프로그램으로 교체해 측정 작업자가 낮은 RSS에서 시작하게 한다.

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

PSS와 USS에는 커널이 보관하는 최대값이 없다. 500 ms마다 `smaps_rollup`을 읽은 값 중 최대를 기록하므로 키에 `_sampled`가 붙는다. `smaps_rollup`은 페이지 테이블을 순회해 비용이 크기 때문에 RSS의 50 ms 주기보다 느리게 읽는다.

## 시나리오 실행

Linux Release 빌드와 Python 개발 의존성이 필요하다.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

M3.5 시나리오 전체를 실행한다.

```sh
.venv/bin/python bench/runner.py \
  --milestone M3.5 \
  --out bench/results/local.json
```

시나리오 하나만 실행하려면 필터를 사용한다.

```sh
.venv/bin/python bench/runner.py \
  --milestone M3.5 \
  --filter single_short_f16 \
  --out bench/results/short.json
```

`runner.py`는 `bench/scenarios.yaml`에서 해당 마일스톤을 포함한 시나리오만 선택한다. 각 시나리오는 별도 `nanoembed-bench` 프로세스로 실행되고 집계 JSON의 `scenarios` 아래에 저장된다.

## 기준 측정값과 비교

```sh
.venv/bin/python bench/compare.py \
  bench/baseline/M3.json \
  bench/results/local.json
```

비교기는 먼저 실행 환경을 확인한다.

- Linux 커널
- CPU 모델
- 논리 CPU 수
- 페이지 크기

환경이 다르면 같은 코드라도 지연 시간과 메모리가 달라질 수 있다. 기본 모드는 경고 후 표를 출력하고, `--strict`는 환경 불일치에서 종료 코드 2를 반환한다.

현재 비교 정책은 다음과 같다.

- p50, p90, p99 지연 시간의 15% 초과 악화는 회귀로 보고한다.
- 15% 초과 개선도 측정 조건 변화인지 검토할 수 있도록 별도 개선 항목으로 보고한다.
- 전체 생애 최대 RSS의 5% 초과 증가는 회귀로 보고한다.
- PSS, USS와 page fault는 표에 출력하지만 자동 실패 조건으로 사용하지 않는다.
- 기준에 있던 시나리오가 현재 결과에서 사라지면 불완전한 실행으로 보고한다.
- `--strict`에서만 회귀가 비정상 종료 코드 1을 만든다.

지연 시간이 크게 좋아졌다고 기준값을 즉시 바꾸면 안 된다. 입력 개수, 스레드 수, 풀링, 모델과 측정 방식이 같았는지 먼저 확인한다.

## `bench_selftest`

측정 도구 자체가 잘못되면 모델 수치도 믿을 수 없다. Linux CTest의 `bench_selftest`는 GGUF 없이 합성 메모리 할당을 사용한다.

- 부모가 64 MiB를 먼저 사용해도 작업 프로세스 시작 RSS에 섞이지 않는지 확인한다.
- `clear_refs`로 측정 구간 최대 RSS를 재설정할 수 있는지 확인한다.
- 작업 프로세스가 할당한 64 MiB만큼 RSS와 USS가 증가하는지 확인한다.
- 충분한 RSS와 `smaps_rollup` 샘플이 수집되는지 확인한다.

이 테스트는 모델 성능을 재는 테스트가 아니다. 측정 방법이 의도한 구간과 메모리를 관찰하는지 확인한다.

## 이 장의 핵심 정리

- 토큰, 레이어와 최종 출력 기준을 나눠야 실패 위치를 좁힐 수 있다.
- 모델이 없으면 로컬 일부 검사는 건너뛰지만 CI는 모델을 필수로 요구한다.
- `forward` 테스트는 각 블록에 Hugging Face 입력을 직접 넣어 오차 누적을 분리한다.
- `seq_len`은 큰 버퍼 재사용과 장문 안전성을 확인한다.
- 벤치마크는 `fork+exec` 작업 프로세스를 바깥에서 관찰한다.
- 성능 결과는 같은 환경에서 얻은 기준 측정값과 비교해야 한다.
- RSS, PSS와 USS는 서로 다른 메모리 질문에 답한다.

[← 06. API와 소유권](06-api-and-ownership.md) | [08. 코드 규칙과 확장 →](08-conventions-and-extending.md)
