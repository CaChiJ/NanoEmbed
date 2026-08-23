[← 04. 모델 로드 경로](04-load-path.md) | [06. API와 소유권 →](06-api-and-ownership.md)

# 05. 한 번의 순전파

이 장은 이미 로드된 모델로 문장 하나를 임베딩하는 실제 코드 경로를 따라간다. 모델 계산 원리는 03장에서 설명했다. 여기서는 그 계산이 객체, 메모리와 ggml 호출로 어떻게 이어지는지 본다.

## 공개 API에서 `Embedder`까지

호출자는 다음 함수를 사용한다.

```c
int nanoembed_embed(
    nanoembed_context * ctx,
    const char * text,
    float * out);
```

[`c_api.cpp`](../../../src/api/c_api.cpp)은 세 포인터가 `nullptr`인지 먼저 확인한다. 정상이라면 컨텍스트가 가진 모델, 실행 버퍼와 설정을 `Embedder::embed()`에 전달한다.

```text
nanoembed_embed
  → ctx->model->embedder.embed(
        ctx->scratch,
        text,
        ctx->cfg,
        out)
```

각 인자의 역할은 다음과 같다.

| 인자 | 역할 |
|---|---|
| `Embedder` | 읽기 전용 가중치, 모델 구조, 토크나이저 |
| `ComputeScratch` | 이 컨텍스트의 CPU 백엔드와 재사용 그래프 버퍼 |
| `EmbedderConfig` | 스레드 수, 길이 상한, 풀링, 정규화 |
| `text` | 입력 문자열 |
| `out` | 호출자가 준비한 `n_embed`개 `float` 배열 |

## 컨텍스트 생성 때 먼저 예약하는 메모리

`nanoembed_new_context()`는 설정을 검사한 뒤 `ComputeScratch`를 만든다. `ComputeScratch`에는 세 가지가 들어 있다.

- CPU 백엔드
- `ggml_gallocr` 그래프 할당기
- 텐서 메타데이터용 작은 연속 메모리 영역

그다음 `Embedder::reserve()`를 호출해 설정의 `max_seq_len`에 필요한 활성값 버퍼 크기를 미리 계산한다.

```text
max_seq_len=512인 가상 입력 그래프 구성
  → ggml_gallocr_reserve()
  → 가장 큰 동시 활성값 사용량 계산
  → 해당 크기의 재사용 버퍼 확보
```

실제 추론 전에 예약하는 이유는 두 가지다.

1. 요청한 최대 길이를 처리할 메모리가 부족하면 첫 호출 도중이 아니라 컨텍스트 생성에서 실패한다.
2. 짧은 입력과 긴 입력을 번갈아 실행해도 큰 버퍼를 계속 다시 만들지 않는다.

요청 길이는 모델 자체 최대 길이보다 커질 수 없다. BERT 모델의 상한이 512인데 사용자가 100,000을 요청하면 예약 길이를 512로 제한한다.

## 스레드 수 결정

`n_threads`가 양수면 그 값을 그대로 사용한다. 0이면 자동 선택한다.

일반 CPU에서는 `std::thread::hardware_concurrency()`를 사용하고 값을 알 수 없으면 4를 사용한다.

Apple의 성능 코어와 효율 코어가 섞인 CPU에서는 `hw.perflevel0.logicalcpu`를 먼저 읽는다. ggml CPU 작업자는 그래프 노드마다 서로 기다릴 수 있다. 느린 효율 코어가 섞이면 전체 노드가 가장 느린 작업자를 기다려 오히려 처리 시간이 늘 수 있기 때문이다.

이 선택은 모든 머신에서 항상 최적이라는 뜻이 아니다. 벤치마크에서는 머신 간 변수를 줄이기 위해 시나리오에 스레드 수를 명시한다.

## 입력 길이와 토큰화

설정의 최대 길이와 모델 자체 최대 길이 중 작은 값을 사용한다.

```text
effective_limit = min(config.max_seq_len, model.max_seq_len)
```

`config.max_seq_len`이 내부 호출에서 0이면 모델 최대 길이를 사용한다. 공개 C API의 컨텍스트 생성은 0 이하 값을 거부하므로 일반 사용자 경로에서는 양수다.

토크나이저는 이 상한 안에서 `[CLS]`, 본문 조각과 `[SEP]`을 만든다.

```cpp
const std::vector<int> ids = tokenizer->encode(text, limit);
const int64_t S = ids.size();
```

`S`는 특수 토큰을 포함한 실제 입력 길이다.

혹시 실제 길이가 현재 예약보다 크면 `reserve()`를 한 번 더 호출해 버퍼를 키운다. 예약은 단조롭게 증가하며 같은 크기 이하 요청에서는 아무 작업도 하지 않는다.

## 그래프 메타데이터 컨텍스트 만들기

`ComputeScratch::new_meta_ctx()`는 `no_alloc=true`인 `ggml_context`를 만든다.

이 컨텍스트에는 다음 정보만 들어간다.

- 입력 텐서 설명
- 모델 연산의 출력 텐서 설명
- 각 텐서가 어떤 연산과 연결되는지
- 최종 `ggml_cgraph`

Q, K, V와 FFN 중간 결과의 실제 숫자는 아직 저장하지 않는다. 실제 데이터 공간은 그래프 전체를 만든 다음 `ggml_gallocr`가 배치한다.

메타데이터 영역은 최대 그래프 텐서 수를 4,096개로 잡는다. BERT 12개 블록의 실제 노드 수보다 여유가 있다. 이 영역도 0으로 초기화되어 RSS에 포함되므로 무한히 크게 잡지 않는다.

## 입력 텐서 만들기

`Embedder::Impl::build_graph()`는 모델 구조가 요구하는 입력을 확인한다.

```text
BERT
  token_ids 필요
  position_ids 필요
  type_ids 필요

gemma3
  token_ids 필요
  position_ids 필요: RoPE가 위치 램프를 입력으로 받는다
  type_ids 불필요
```

두 계열이 `position_ids`를 같이 쓰는 것은 우연이 아니다. 값이 둘 다 `0..S-1`
램프이고, BERT는 그것으로 학습된 위치 표를 조회하며 `gemma3`는 그것을 RoPE에
넘긴다. 덕분에 두 번째 계열을 추가하면서 입력 배선을 건드릴 필요가 없었다.
causal 마스크도 마찬가지로 그래프 안에서 만들기 때문에 새 입력 텐서가 없다.

현재 BERT 입력은 모두 I32 `[S, 1]` 텐서다.

각 입력에는 `ggml_set_input()`을 호출한다. 그래프 할당 뒤 외부에서 값을 쓸 텐서이므로, 그 공간을 아직 사용 중인 다른 텐서와 겹치지 않게 하기 위한 표시다.

## 모델 구조에 순전파 그래프 요청하기

입력 텐서를 만들면 `ModelArch::build_graph()`을 호출한다.

현재 `BertModelArch`는 다음 순서로 공통 BERT 그래프 빌더를 연결한다.

```text
build_embed_layer(inputs, embedding_weights)
  → build_encoder_block(layer 0 weights)
  → build_encoder_block(layer 1 weights)
  → ...
  → build_encoder_block(layer 11 weights)
```

각 함수는 숫자를 즉시 계산하지 않는다. 입력과 가중치를 참조하는 새 `ggml_tensor` 노드를 반환한다.

모델 구조가 반환한 `[H, S, B]` 출력 뒤에는 공통 풀링과 선택적 L2 정규화를 연결한다.

```text
인코더 출력 [H, S, B]
  → Mean / CLS / Last 풀링
  → [H, B]
  → 선택적 L2 정규화
  → 최종 출력 [H, B]
```

최종 출력에는 `ggml_set_output()`을 호출한다. 계산이 끝난 뒤 호출자 배열로 읽을 때까지 데이터가 유지되어야 하기 때문이다.

## 최종 출력에서 실행 그래프 수집하기

최종 출력 텐서 하나를 시작점으로 `ggml_build_forward_expand()`를 호출한다.

ggml은 출력이 의존하는 텐서를 역으로 따라간다.

```text
최종 정규화
  ← 풀링
  ← 11번 BERT 블록
  ← ...
  ← 0번 BERT 블록
  ← 입력 임베딩
  ← 토큰·위치·타입 ID
```

그 결과가 실제 실행 순서를 가진 `ggml_cgraph`다.

## 활성값 버퍼 배치

이제 `ggml_gallocr_alloc_graph()`를 호출한다.

할당기는 각 텐서가 처음 만들어지는 시점과 마지막으로 사용되는 시점을 본다. 동시에 살아 있지 않은 텐서는 같은 버퍼 구간을 사용할 수 있다.

```text
초기 M3
  텐서 A 공간 | 텐서 B 공간 | 텐서 C 공간 | ...
  계산이 끝난 공간도 그래프 종료까지 재사용하지 않음

현재
  텐서 A 공간을 마지막 사용 후 텐서 C가 재사용
  그래프 전체에서 동시에 필요한 최대 크기만 유지
```

컨텍스트 생성 때 이미 충분한 크기를 예약했으므로 정상 경로에서는 기존 버퍼 안에 배치된다.

## 입력 데이터 복사

그래프 할당 뒤 입력 텐서는 실제 데이터 주소를 가진다.

### 토큰 ID

토크나이저가 만든 정수 배열을 그대로 복사한다.

```text
[CLS], word pieces, [SEP]
  → I32 token_ids[S, 1]
```

### 위치 ID

0부터 `S-1`까지 순서대로 만든다.

```text
0, 1, 2, ..., S-1
```

### 타입 ID

현재 단일 문장 입력이므로 모두 0이다.

```text
0, 0, 0, ..., 0
```

모델 구조가 입력을 요구하지 않으면 해당 텐서는 `nullptr`이고 복사도 생략한다.

## CPU 계산 실행

컨텍스트의 CPU 백엔드에 스레드 수를 설정한 뒤 그래프를 실행한다.

```cpp
ggml_backend_cpu_set_n_threads(sc.backend, n_threads);
ggml_status st = ggml_backend_graph_compute(sc.backend, graph);
```

여기서 처음으로 임베딩 조회, 행렬곱, softmax, 정규화와 풀링의 실제 숫자 계산이 일어난다.

상태가 `GGML_STATUS_SUCCESS`가 아니면 C++ 예외를 던진다. C API 경계는 이 예외를 `NANOEMBED_ERR_INTERNAL`과 오류 문자열로 바꾼다.

## 출력 복사와 정리

계산이 성공하면 최종 `[H, 1]` 텐서에서 `H`개 F32 값을 호출자의 `out` 배열로 복사한다.

```text
ggml 출력 텐서
  → float out[384]
```

그다음 메타데이터용 `ggml_context`를 해제한다.

해제되는 것과 유지되는 것을 구분해야 한다.

| 호출 뒤 해제 | 다음 호출까지 유지 |
|---|---|
| 이번 그래프의 텐서 메타데이터 | 모델 가중치 |
| 이번 `ggml_cgraph` 설명 | 토크나이저 |
| 위치·타입 ID 임시 `std::vector` | CPU 백엔드 |
| | `ggml_gallocr`와 활성값 버퍼 |

활성값의 이전 숫자가 버퍼에 남아 있을 수 있지만 다음 그래프가 필요한 구간을 다시 쓴다. `seq_len` 테스트는 긴 입력과 짧은 입력을 번갈아 실행해 이전 결과가 다음 호출을 오염시키지 않는지 확인한다.

## BERT 그래프 빌더와 코드 대응

| 모델 계산 | 함수 | 입력과 출력 |
|---|---|---|
| 토큰·위치·타입 임베딩 | `build_embed_layer` | ID `[S,B]` → `[H,S,B]` |
| 셀프 어텐션 | `build_attention_block` | `[H,S,B]` → `[H,S,B]` |
| FFN | `build_ffn_block` | `[H,S,B]` → `[H,S,B]` |
| BERT 블록 | `build_encoder_block` | 어텐션과 FFN 연결 |
| Mean/CLS/Last 풀링 | `build_pool` | `[H,S,B]` → `[H,B]` |
| L2 정규화 | `build_l2_normalize` | `[H,B]` → `[H,B]` |

각 빌더는 실행 상태를 소유하지 않는다. 필요한 입력, 가중치와 컨텍스트를 받아 그래프 노드를 반환한다. 이 경계 덕분에 레이어별 테스트가 가능하고, 앞으로 스트리밍 실행에서도 같은 계산 빌더를 재사용할 수 있다.

## 직접 실행해 보기

```sh
# 기본값: Mean 풀링과 L2 정규화
echo "hello world" \
  | ./build/bin/nanoembed-cli models/bge-small-en-v1.5-f16.gguf

# CLS 풀링
echo "hello world" \
  | ./build/bin/nanoembed-cli models/bge-small-en-v1.5-f16.gguf --cls

# 정규화 끄기
echo "hello world" \
  | ./build/bin/nanoembed-cli models/bge-small-en-v1.5-f16.gguf --no-normalize

# 스레드 수 고정
echo "hello world" \
  | ./build/bin/nanoembed-cli models/bge-small-en-v1.5-f16.gguf --threads 4
```

정규화를 끄면 출력 벡터의 L2 길이가 1이 아닐 수 있다. CLS와 Mean은 같은 토큰별 인코더 출력을 서로 다른 방식으로 문장 하나의 벡터로 줄인다.

## 이 장의 핵심 정리

- 모델은 읽기 전용 가중치와 토크나이저를 제공하고 컨텍스트는 변경 가능한 실행 버퍼를 제공한다.
- 컨텍스트 생성 때 최대 입력 길이의 그래프 버퍼를 미리 예약한다.
- 순전파는 그래프 구성, 데이터 공간 배치, 입력 복사, CPU 실행, 출력 복사 순서로 진행된다.
- `ggml_gallocr`는 중간 텐서 수명을 분석해 같은 공간을 재사용한다.
- 호출 뒤 그래프 설명은 버리지만 큰 활성값 버퍼는 다음 호출을 위해 유지한다.
- 모델별 인코더 그래프와 공통 풀링·정규화가 분리되어 있다.

[← 04. 모델 로드 경로](04-load-path.md) | [06. API와 소유권 →](06-api-and-ownership.md)
