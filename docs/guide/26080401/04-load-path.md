[← 03. BERT 임베딩 모델](03-model-theory.md) | [05. 한 번의 순전파 →](05-forward-path.md)

# 04. 모델 로드 경로

이 장은 `nanoembed_load_model()`이 GGUF 파일을 받아 실행 가능한 모델 객체를 만드는 과정을 설명한다.

모델 로드에는 세 가지 목적이 있다.

1. 파일이 NanoEmbed가 이해하는 GGUF인지 확인한다.
2. 모델 구조와 토크나이저 종류를 선택한다.
3. 계산에 사용할 실제 가중치 텐서를 연결한다.

잘못된 파일은 첫 추론까지 기다리지 않고 로드 시점에 구체적인 오류로 거부한다.

## 전체 호출 흐름

```text
nanoembed_load_model(path)
  │
  ▼
nanoembed_model 생성
  │
  ▼
Embedder(path)
  ├─ create_model_arch(path)
  │    ├─ general.architecture 읽기
  │    └─ bert이면 BertModelArch(path)
  │          └─ scan_gguf(path)
  │               메타데이터와 모든 BERT 텐서 검사
  │
  ├─ 같은 GGUF를 실제 가중치 데이터와 함께 열기
  ├─ create_tokenizer(gguf)
  │    └─ tokenizer.ggml.model이 bert이면 WordPiece 구성
  └─ arch->bind_weights(model_ctx)
       이름으로 실제 ggml_tensor 포인터 연결
```

파일은 여러 함수에서 열리지만 목적이 다르다. 먼저 구조를 선택하고 검사한 뒤, 계산용 가중치를 소유하는 컨텍스트를 별도로 만든다.

## C API에서 C++ 객체로 넘어가기

공개 API의 모델은 내부 구조를 노출하지 않는 불투명 타입이다.

```c
typedef struct nanoembed_model nanoembed_model;
```

구현 파일에서는 이 구조체가 `Embedder` 하나를 가진다.

```cpp
struct nanoembed_model {
    explicit nanoembed_model(const std::string & path) : embedder(path) {}
    nanoembed::Embedder embedder;
};
```

`new nanoembed_model(path)` 중에 예외가 나면 `nanoembed_load_model()`이 예외를 잡는다. 오류 메시지를 thread-local 버퍼에 복사하고 `nullptr`을 반환한다.

```text
C++ 내부 예외
  → C API 경계에서 catch
  → nanoembed_last_error()에 문자열 저장
  → C 호출자에게 nullptr 반환
```

C 호출자는 예외 처리 문법을 알 필요가 없다.

## 1단계: 모델 구조 이름만 먼저 읽는다

GGUF의 `general.architecture`는 모델 계산 구조를 나타낸다.

```text
general.architecture = "bert"
```

[`create_model_arch()`](../../../src/arch/registry.cpp)은 이 문자열을 먼저 읽는다. 이 단계에서는 특정 모델의 텐서 이름을 아직 가정하지 않는다.

이 순서가 필요한 이유는 `gemma3` 파일로 바로 확인할 수 있다. `gemma3`에는 BERT의
`position_embd.weight`도, bias가 있는 Q·K·V 텐서도 없다. 구조 이름을 보기 전에
BERT 검사기를 돌리면 “position tensor가 없다”는 오류가 나오는데, 실제 문제는
파일이 손상된 것이 아니라 다른 계열이라는 것이다.

먼저 구조 이름을 읽으면 각 계열을 자기 검사기로 보낼 수 있고, 지원하지 않는
계열에는 정확한 오류를 만들 수 있다.

```text
architecture 'eurobert' (jina-embeddings-v5-text-nano) is recognized but not implemented
```

현재 레지스트리의 동작은 다음과 같다.

| `general.architecture` | 결과 |
|---|---|
| `bert` | `BertModelArch` 생성 |
| `gemma3` | `Gemma3ModelArch` 생성 |
| `eurobert` | 종류를 명시한 미구현 오류 |
| 그 밖의 값 | 지원하지 않는 구조 오류 |

## 2단계: BERT GGUF를 검사한다

`BertModelArch` 생성자는 [`scan_gguf()`](../../../src/gguf_scanner.cpp)을 호출한다.

검사기는 `no_alloc=true`로 GGUF를 연다. 이 모드에서는 텐서의 이름, 자료형, 각 축의 크기와 파일 위치를 읽지만 가중치 데이터 전체를 위한 메모리는 할당하지 않는다.

### 하이퍼파라미터 검사

다음 메타데이터를 읽는다.

| GGUF 키 | 의미 | bge-small 값 |
|---|---|---:|
| `bert.block_count` | 인코더 레이어 수 | 12 |
| `bert.embedding_length` | hidden 차원 H | 384 |
| `bert.attention.head_count` | 어텐션 헤드 수 | 12 |
| `bert.feed_forward_length` | FFN 차원 F | 1,536 |
| `bert.context_length` | 최대 토큰 수 | 512 |
| `bert.attention.layer_norm_epsilon` | LayerNorm epsilon | 모델 값 또는 기본 `1e-12` |
| `tokenizer.ggml.tokens` 길이 | 어휘 크기 V | 30,522 |

값이 양수인지 확인하고 `H`가 헤드 수로 나누어지는지도 검사한다.

```text
head_dim = H / n_head = 384 / 12 = 32
```

이 조건이 맞지 않으면 멀티헤드 어텐션으로 hidden 차원을 균등하게 나눌 수 없다.

### 임베딩 텐서 검사

| 텐서 | 예상 ggml 차원 크기 | 역할 |
|---|---|---|
| `token_embd.weight` | `[H, V]` | 토큰 ID를 벡터로 변환 |
| `position_embd.weight` | `[H, Smax]` | 위치 벡터 |
| `token_types.weight` | `[H, n_types]` | 문장 타입 벡터 |
| `token_embd_norm.weight` | `[H]` | 입력 LayerNorm gain |
| `token_embd_norm.bias` | `[H]` | 입력 LayerNorm bias |

### 레이어 텐서 검사

각 레이어는 어텐션과 FFN 가중치를 가진다. 레이어 번호 0의 예는 다음과 같다.

```text
blk.0.attn_q.weight
blk.0.attn_q.bias
blk.0.attn_k.weight
...
blk.0.ffn_up.weight
blk.0.ffn_down.weight
blk.0.layer_output_norm.weight
```

가중치가 이름만 존재하는지 확인하는 것으로 충분하지 않다. 각 축의 크기도 계산과 맞아야 한다.

| 종류 | 예상 차원 크기 |
|---|---|
| Q·K·V·출력 선형 변환 | `[H, H]` |
| 선형 변환 편향값 | `[H]` |
| FFN up | `[H, F]` |
| FFN up bias | `[F]` |
| FFN down | `[F, H]` |
| FFN down bias | `[H]` |
| 정규화 weight와 bias | `[H]` |

필수 텐서가 없거나 차원 크기가 다르면 `ScanError`를 던진다. 예외 메시지에는 텐서 이름, 예상 크기와 실제 크기가 들어간다.

## `ModelManifest`는 무엇인가

검사 결과는 `ModelManifest`에 정리된다.

- `BertArch`: 레이어 수, hidden 차원 같은 하이퍼파라미터
- 임베딩 단계 `TensorRef`
- 레이어별 Q·K·V, FFN과 정규화 `TensorRef`

`TensorRef`는 실제 가중치를 소유하지 않는다. GGUF 안의 텐서 번호, 자료형, 각 축의 크기, 바이트 수와 파일 위치를 기록한 설명이다.

`ModelManifest`는 BERT 전용 검사 결과다. 모든 모델이 공유하는 공개 계약이 아니다.
`gemma3`처럼 텐서 구성이 다른 모델은 `Gemma3Manifest`와 `scan_gemma3()`로 자기
구조에 맞는 검증을 따로 구현한다. 두 검사기가 공유하는 것은 계열과 무관한 부분,
즉 `gguf_util.h`의 KV 읽기·텐서 조회·차원 검증뿐이다.

`BertModelArch`는 검사 결과에서 공통 하이퍼파라미터를 `ArchParams`로 복사하고 BERT 검사 결과인 manifest를 내부에 보관한다.

## `ScanResult`와 검사 자원의 수명

`scan_gguf()`는 `ScanResult`를 반환한다. 이 객체는 검사에 사용한 `gguf_context`와 메타데이터용 `ggml_context`를 RAII 방식으로 소유한다.

검사 도중 예외가 나면 임시 `unique_ptr`가 두 컨텍스트를 정리한다. 검사가 성공하면 소유권을 `ScanResult`로 옮긴다.

`BertModelArch` 생성자에서는 manifest 내용을 값으로 복사한다. 생성자가 끝나면 임시 `ScanResult`가 검사 컨텍스트를 해제한다. 실제 계산에는 다음 단계에서 새로 여는 가중치 컨텍스트를 사용한다.

## 3단계: 실제 가중치 데이터를 연다

검사를 통과하면 `Embedder`가 같은 파일을 `no_alloc=false`로 다시 연다.

```cpp
gguf_init_params gp;
gp.no_alloc = false;
gp.ctx = &model_ctx;
gguf = gguf_init_from_file(path, gp);
```

이 `model_ctx`의 텐서는 실제 데이터 주소를 가진다. 현재 구현에서는 모델 가중치 전체가 모델 핸들의 수명 동안 메모리에 있다.

같은 파일을 두 번 여는 이유는 역할을 분리하기 위해서다.

| 첫 번째 검사 경로 | 두 번째 실행 경로 |
|---|---|
| 가중치 데이터 불필요 | 실제 가중치 데이터 필요 |
| 잘못된 구조를 구체적으로 보고 | 계산에 사용할 텐서 포인터 제공 |
| 임시 수명 | 모델 핸들과 같은 수명 |

M4에서 `mmap` 기반 스트리밍을 추가하면 두 번째 경로의 소유 방식은 바뀐다. 구조를 먼저 검증한다는 원칙은 유지된다.

## 4단계: 토크나이저를 선택한다

모델 구조를 골랐다고 토크나이저도 자동으로 같은 종류라고 가정하지 않는다.

[`create_tokenizer()`](../../../src/tokenizer/registry.cpp)은 실행용 GGUF의 `tokenizer.ggml.model`을 읽는다.

| 값 | 결과 |
|---|---|
| `bert` | `WordPieceTokenizer` 구성 |
| `gpt2` | byte-level BPE 미구현 오류 |
| 그 밖의 값 | 지원하지 않는 토크나이저 오류 |

WordPiece 생성자는 GGUF에서 다음 정보를 읽는다.

- `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` 토큰 ID
- `tokenizer.ggml.tokens` 문자열 배열

GGUF 변환기가 저장한 토큰 문자열과 Hugging Face WordPiece 표기는 시작 부분 표현이 다르다. NanoEmbed는 로드 시 문자열 표기를 WordPiece 방식으로 정규화하되 토큰 ID는 그대로 유지한다.

## 5단계: 이름을 실제 가중치 포인터에 연결한다

검사기는 필요한 이름과 차원 크기가 맞다는 것을 이미 확인했다. `BertModelArch::bind_weights()`는 실행용 `model_ctx`에서 같은 이름의 `ggml_tensor*`를 찾는다.

```text
"token_embd.weight"
  → EmbedWeights::tok

"blk.0.attn_q.weight"
  → layer_w[0].attn.q_w
```

그래프 빌더는 문자열 이름을 매번 검색하지 않는다. 구조화된 `EmbedWeights`와 `LayerWeights`를 받는다.

이 단계에서도 텐서를 찾지 못하면 예외를 던진다. 정상적인 파일이라면 앞의 검사가 이미 같은 이름을 확인했으므로, 실패는 검사 경로와 실행 경로 사이의 불일치를 뜻한다.

## 로드 시점에는 실행 버퍼를 만들지 않는다

모델 로드는 가중치와 토크나이저를 준비한다. CPU 백엔드와 활성값 버퍼는 컨텍스트를 만들 때 준비한다.

```text
nanoembed_model
  모델 가중치 + 모델 구조 + 토크나이저

nanoembed_context
  실행 설정 + CPU 백엔드 + 그래프 할당기 + 활성값 버퍼
```

이 분리 덕분에 모델 하나를 여러 컨텍스트가 공유할 수 있다. 컨텍스트마다 다른 스레드 수나 최대 입력 길이를 사용할 수도 있다.

## 실패 시나리오

| 문제 | 발견 위치 | 오류의 핵심 정보 |
|---|---|---|
| 파일을 열 수 없음 | 구조 이름 확인 또는 GGUF 열기 | 파일 경로 |
| `general.architecture` 없음 | 모델 레지스트리 | 필수 문자열 키가 없음 |
| `eurobert` 파일 | 모델 레지스트리 | 인식했지만 아직 미구현 |
| 메타데이터 누락 | 해당 계열 검사기 | 누락된 키 이름 |
| hidden 차원과 헤드 수 불일치 | BERT 검사기 | 나누어지지 않음 |
| 쿼리/KV 헤드 수 불일치 | gemma3 검사기 | 배수가 아님 |
| 필수 텐서 누락 | 해당 계열 검사기 | 텐서 이름 |
| 텐서 차원 크기 불일치 | 해당 계열 검사기 | 예상값과 실제값 |
| 토크나이저 종류 미지원 | 토크나이저 레지스트리 | GGUF에 기록된 종류 |
| 특수 토큰 ID 누락 | WordPiece 생성 | 누락된 토크나이저 키 |

## 직접 확인하기

기본 출력은 GGUF 메타데이터와 구조화된 BERT 검사 결과를 보여 준다.

```sh
./build/bin/nanoembed-inspect \
  models/bge-small-en-v1.5-f16.gguf
```

모든 원시 텐서 이름과 자료형까지 보려면 `--tensors`를 사용한다.

```sh
./build/bin/nanoembed-inspect \
  models/bge-small-en-v1.5-f16.gguf --tensors
```

`--graph`는 검사만 하지 않고 실제 모델과 실행 버퍼를 만든다. 따라서 가중치 로드와 길이 512 활성값 버퍼의 비용을 확인할 수 있다.

```sh
./build/bin/nanoembed-inspect \
  models/bge-small-en-v1.5-f16.gguf --graph
```

## 이 장의 핵심 정리

- 모델 구조 이름을 먼저 읽어 잘못된 BERT 오류 대신 실제 미지원 구조를 보고한다.
- BERT 검사기는 가중치를 계산에 쓰기 전에 모든 메타데이터와 텐서 차원 크기를 확인한다.
- manifest라는 검사 결과는 실제 가중치가 아니라 텐서의 설명을 담는다.
- 실행용 GGUF 컨텍스트는 실제 가중치 데이터를 소유한다.
- 모델 구조와 토크나이저는 서로 다른 GGUF 키로 독립적으로 선택한다.
- 실행 버퍼는 모델이 아니라 각 컨텍스트가 소유한다.

[← 03. BERT 임베딩 모델](03-model-theory.md) | [05. 한 번의 순전파 →](05-forward-path.md)
