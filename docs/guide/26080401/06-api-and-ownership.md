[← 05. 한 번의 순전파](05-forward-path.md) | [07. 테스트와 벤치마크 →](07-tests-bench-workflow.md)

# 06. 공개 API, 객체 수명과 오류 처리

이 장은 NanoEmbed를 사용하는 코드가 지켜야 할 규칙을 설명한다. 모델과 컨텍스트를 나눈 이유, 어떤 객체가 메모리를 소유하는지, 여러 스레드에서 무엇을 공유할 수 있는지가 핵심이다.

## 공개 API가 C인 이유

라이브러리 내부는 C++17로 작성되어 있다. 외부에는 C ABI를 제공한다.

C ABI는 다음 장점이 있다.

- C와 C++에서 직접 호출할 수 있다.
- Node.js, Python, Rust 같은 다른 언어가 호출할 수 있는 공통 경계를 제공한다.
- C++ 컴파일러의 이름 장식과 표준 라이브러리 ABI 차이를 공개 경계 밖에 둘 수 있다.
- 내부 클래스를 바꾸더라도 공개 함수 모양을 유지할 수 있다.

공개 선언은 [`include/nanoembed/nanoembed.h`](../../../include/nanoembed/nanoembed.h)에만 있다. ggml 타입과 `Embedder`는 외부에 노출하지 않는다.

## 가장 작은 사용 흐름

```c
#include <nanoembed/nanoembed.h>

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    nanoembed_model * model =
        nanoembed_load_model("models/bge-small-en-v1.5-f16.gguf");
    if (model == NULL) {
        fprintf(stderr, "model load failed: %s\n", nanoembed_last_error());
        return 1;
    }

    nanoembed_context_params params = nanoembed_context_default_params();
    params.pooling = NANOEMBED_POOL_CLS;

    nanoembed_context * ctx = nanoembed_new_context(model, params);
    if (ctx == NULL) {
        fprintf(stderr, "context creation failed: %s\n", nanoembed_last_error());
        nanoembed_free_model(model);
        return 1;
    }

    const int n_embed = nanoembed_n_embed(model);
    if (n_embed <= 0) {
        fprintf(stderr, "metadata failed: %s\n", nanoembed_last_error());
        nanoembed_free_context(ctx);
        nanoembed_free_model(model);
        return 1;
    }

    float * out = (float *) malloc((size_t) n_embed * sizeof(float));
    if (out == NULL) {
        nanoembed_free_context(ctx);
        nanoembed_free_model(model);
        return 1;
    }

    const int rc = nanoembed_embed(ctx, "hello world", out);
    if (rc != NANOEMBED_OK) {
        fprintf(stderr, "embed failed: %s\n", nanoembed_last_error());
    }

    free(out);
    nanoembed_free_context(ctx);
    nanoembed_free_model(model);
    return rc == NANOEMBED_OK ? 0 : 1;
}
```

해제 순서가 중요하다. 컨텍스트가 모델을 참조하므로 모든 컨텍스트를 먼저 해제하고 모델을 마지막에 해제한다.

## 불투명 핸들

공개 헤더에는 구조체 내부가 보이지 않는다.

```c
typedef struct nanoembed_model   nanoembed_model;
typedef struct nanoembed_context nanoembed_context;
```

호출자는 포인터를 저장하고 함수에 전달할 수 있지만 필드에는 접근할 수 없다.

구현에서는 두 핸들이 다음 C++ 객체를 감싼다.

```text
nanoembed_model
  └─ Embedder
       ├─ GGUF와 가중치 컨텍스트
       ├─ ModelArch
       └─ Tokenizer

nanoembed_context
  ├─ nanoembed_model*       모델을 빌려 쓰는 포인터
  ├─ EmbedderConfig         실행 설정
  └─ ComputeScratch         CPU 백엔드와 실행 버퍼
```

핸들 내부를 감추면 `ComputeScratch`의 필드나 모델 레지스트리 구조를 바꿔도 공개 헤더를 사용하는 프로그램을 최소한으로 영향을 줄 수 있다.

## 모델과 컨텍스트를 나눈 이유

모델과 실행 상태의 수명이 다르다.

### 모델

모델은 로드 후 읽기 전용으로 사용하는 데이터를 소유한다.

- GGUF 핸들
- 실제 가중치 텐서
- 모델 구조 구현
- 토크나이저와 어휘

이 데이터는 여러 요청이 함께 읽을 수 있다. 모델 로드는 가중치와 큰 어휘를 준비하므로 컨텍스트마다 반복하면 낭비다.

### 컨텍스트

컨텍스트는 호출 중에 바뀌는 상태를 소유한다.

- 스레드 수와 입력 길이 상한
- 풀링과 정규화 설정
- CPU 백엔드
- 그래프 할당기
- 입력과 중간 활성값을 저장하는 재사용 버퍼

한 문장을 계산하는 동안 입력 ID와 중간 결과가 이 버퍼에 계속 기록된다. 따라서 같은 컨텍스트를 동시에 두 번 실행할 수 없다.

## 객체 수명 규칙

```text
model 생성
  ├─ context A 생성
  │    ├─ embed 호출들
  │    └─ context A 해제
  ├─ context B 생성
  │    ├─ embed 호출들
  │    └─ context B 해제
  └─ model 해제
```

지켜야 할 규칙은 다음과 같다.

1. 모델이 살아 있는 동안에만 그 모델로 만든 컨텍스트를 사용한다.
2. 모델을 해제하기 전에 관련 컨텍스트를 모두 해제한다.
3. 한 컨텍스트에서는 한 시점에 하나의 embed 호출만 실행한다.
4. 서로 다른 컨텍스트는 같은 모델을 공유하면서 동시에 실행할 수 있다.
5. `nanoembed_free_model(NULL)`과 `nanoembed_free_context(NULL)`은 안전한 no-op이다.

모델은 컨텍스트의 존재를 추적하지 않는다. 모델을 먼저 해제하면 컨텍스트의 `model` 포인터가 유효하지 않게 된다. 이 순서는 호출자가 책임진다.

## 컨텍스트 설정

항상 `nanoembed_context_default_params()`로 구조체 전체를 초기화한 뒤 필요한 값만 바꾼다.

```c
nanoembed_context_params p = nanoembed_context_default_params();
p.n_threads = 4;
p.max_seq_len = 256;
p.pooling = NANOEMBED_POOL_CLS;
```

현재 기본값과 의미는 다음과 같다.

| 필드 | 기본값 | 의미 |
|---|---:|---|
| `n_threads` | 0 | 자동 선택. 양수면 해당 CPU 스레드 수 사용 |
| `max_batch` | 64 | 양수여야 함. 실제 배치 분할은 M5 예정 |
| `max_seq_len` | 512 | 2 이상. 토큰화 상한과 활성값 버퍼 예약 길이 |
| `use_streaming` | 0 | 1은 아직 지원하지 않아 컨텍스트 생성 실패 |
| `pooling` | Model default | 모델 기본값 또는 Mean/CLS/LAST |
| `normalize` | 1 | 0이면 L2 정규화 생략 |

### `max_seq_len`과 모델 상한

설정값이 모델의 최대 문맥 길이보다 크더라도 실제 예약과 토큰화는 모델 상한으로 제한한다.

```text
요청 max_seq_len = 100000
모델 최대 문맥 길이 = 512
실제 상한 = 512
```

위치 임베딩 표 밖을 읽는 일을 막기 위한 규칙이다.

긴 문맥 모델의 전체 상한을 무조건 예약하지 않는 것도 중요하다. 어텐션 점수 텐서는
토큰 수의 제곱에 비례한다. 문맥 32,768인 Harrier를 기본값으로 전부 예약하면 현실적으로
감당할 수 없는 활성값 공간이 필요하다. 그래서 공개 API 기본 상한은 512다.

## 출력 버퍼 소유권

NanoEmbed는 최종 출력 버퍼를 할당하지 않는다. 호출자가 충분한 공간을 준비한다.

단일 문장은 `nanoembed_n_embed(model)`개 `float`가 필요하다.

```text
out 크기 = H
```

여러 문장은 문장 수를 곱한다.

```text
out 크기 = n_texts × H

문장 i의 시작 위치 = out + i × H
```

API에는 버퍼 길이 인자가 없으므로 너무 작은 버퍼를 넘기면 라이브러리가 알아낼 수 없다. 올바른 크기를 확보하는 것은 호출자의 책임이다.

## 여러 문장 API의 현재 동작

```c
int nanoembed_embed_batch(
    nanoembed_context * ctx,
    const char * const * texts,
    int n_texts,
    float * out);
```

현재는 다음과 동일하게 한 문장 경로를 반복한다.

```text
for i in 0..n_texts-1:
    embed(texts[i], out + i * H)
```

한 그래프에 여러 문장을 넣는 실제 배치는 아니다. M5에서 내부 구현을 레이어 단위 배치로 바꿔도 공개 함수 모양과 출력 배치는 유지한다.

`n_texts`가 음수이거나 배열 안에 `nullptr` 문장이 있으면 오류다. `n_texts == 0`은 아무 계산 없이 성공한다.

## 반환값과 오류 메시지

함수 종류에 따라 실패 표현이 다르다.

| 함수 종류 | 성공 | 실패 |
|---|---|---|
| 포인터 반환 | 유효한 핸들 | `NULL` |
| 상태 반환 | `NANOEMBED_OK`(0) | 음수 상태 코드 |
| 차원 반환 | 양수 값 | 음수 상태 코드 |

상태 코드의 주요 의미는 다음과 같다.

| 코드 | 의미 |
|---|---|
| `NANOEMBED_ERR_INVALID_ARG` | `nullptr`, 음수 개수, 잘못된 설정 |
| `NANOEMBED_ERR_FILE` | 파일 접근 문제를 위한 공개 코드 |
| `NANOEMBED_ERR_FORMAT` | 잘못된 모델 형식을 위한 공개 코드 |
| `NANOEMBED_ERR_ARCH` | 지원하지 않는 모델 구조를 위한 공개 코드 |
| `NANOEMBED_ERR_TENSOR` | 누락되거나 잘못된 텐서를 위한 공개 코드 |
| `NANOEMBED_ERR_TOKENIZE` | 토큰화 실패를 위한 공개 코드 |
| `NANOEMBED_ERR_OOM` | 메모리 부족을 위한 공개 코드 |
| `NANOEMBED_ERR_NOT_IMPL` | 아직 구현하지 않은 기능을 위한 공개 코드 |
| `NANOEMBED_ERR_INTERNAL` | 현재 C++ 예외 경로의 일반 실패 코드 |

현재 구현은 많은 내부 예외를 세부 공개 코드로 나누지 않고 `NANOEMBED_ERR_INTERNAL`로 변환한다. 구체적인 원인은 `nanoembed_last_error()` 문자열에서 확인한다.

## `nanoembed_last_error()`의 수명

오류 메시지는 스레드마다 별도 512바이트 버퍼에 저장된다.

```c
const char * message = nanoembed_last_error();
```

반환된 포인터는 다음 NanoEmbed 호출이 같은 스레드의 버퍼를 바꾸기 전까지만 유효하다. 오래 보관해야 한다면 호출자가 문자열을 복사한다.

성공한 공개 호출은 오류 버퍼를 비운다. 따라서 실패 직후 바로 메시지를 읽어야 한다.

## C++ 예외가 C 경계를 넘지 않게 한다

내부에서는 `ScanError`, `TokenizerError`, `std::runtime_error` 같은 예외를 사용한다. 공개 함수는 다음 패턴으로 예외를 잡는다.

```text
try
  → 내부 C++ 동작
catch std::exception
  → e.what() 저장
  → 실패 값 반환
catch ...
  → "unknown internal error" 저장
  → 실패 값 반환
```

C++ 예외가 `extern "C"` 함수 밖으로 나가면 다른 언어와 ABI에서 처리할 수 없다. 모든 공개 진입점이 경계가 된다.

## PImpl과 내부 헤더

`Embedder`와 `ComputeScratch`는 PImpl 패턴을 사용한다.

```cpp
class Embedder {
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

헤더는 `Impl`의 존재만 알고 실제 필드는 `.cpp`에 둔다. 이 구조에는 다음 효과가 있다.

- `embedder.h`를 포함해도 ggml 헤더를 함께 파싱하지 않는다.
- 내부 필드 변경이 헤더 사용자에게 전파되지 않는다.
- ggml 포인터의 해제 순서를 구현 파일 한곳에서 관리한다.

PImpl은 메모리 소유권을 자동으로 올바르게 만드는 기능은 아니다. 소멸자에서 GGUF, ggml 컨텍스트, 백엔드와 할당기를 올바른 순서로 해제하는 코드는 여전히 필요하다.

## 이 장의 핵심 정리

- 모델은 읽기 전용 가중치와 토크나이저를 소유한다.
- 컨텍스트는 설정과 변경 가능한 실행 버퍼를 소유한다.
- 같은 모델을 공유하는 서로 다른 컨텍스트는 동시에 사용할 수 있다.
- 같은 컨텍스트를 동시에 호출하면 안 된다.
- 컨텍스트를 먼저 해제하고 모델을 마지막에 해제한다.
- 출력 버퍼 크기는 호출자가 계산한다.
- 내부 예외는 C API 경계에서 상태 값과 thread-local 문자열로 변환된다.

[← 05. 한 번의 순전파](05-forward-path.md) | [07. 테스트와 벤치마크 →](07-tests-bench-workflow.md)
