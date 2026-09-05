[← 00. 가이드 목차](00-index.md) | [02. GGML과 GGUF 기초 →](02-ggml-primer.md)

# 01. 프로젝트 한눈에 보기

이 장에서는 세부 계산보다 프로젝트의 경계를 먼저 잡는다. NanoEmbed가 어떤 문제를 풀려는지, 사용자 호출이 어느 파일을 지나는지, 현재 소스가 어떤 책임으로 나뉘는지를 설명한다.

## NanoEmbed가 해결하려는 문제

텍스트 임베딩 모델은 문장을 고정 길이 숫자 벡터로 바꾼다. 두 문장의 벡터가 가까우면 의미도 비슷하다고 볼 수 있다. 검색, 추천, 중복 탐지와 RAG의 검색 단계에서 이 성질을 사용한다.

Python 서버에서는 `sentence-transformers` 같은 프레임워크로 쉽게 임베딩을 만들 수 있다. NanoEmbed의 조건은 다르다.

- C++에서 직접 호출할 수 있어야 한다.
- Python 런타임에 의존하지 않아야 한다.
- 메모리가 제한된 장치에서도 실행할 수 있어야 한다.
- 메모리 최적화 전후의 결과가 같은지 검증할 수 있어야 한다.

첫 번째 모델인 `bge-small-en-v1.5`의 F16 가중치는 약 64 MiB다. 현재 구현은 이 가중치를 모두 메모리에 올린다. 앞으로는 한 번에 한 인코더 레이어의 가중치만 상주시켜 전체 RSS를 줄일 예정이다.

여기서 RSS는 프로세스가 현재 실제 RAM에 올려 둔 메모리 크기다. 파일 크기나 가상 주소 공간보다 장치의 실제 메모리 부담을 직접 보여 준다.

## 요청 하나가 지나가는 경로

공개 API에서 `nanoembed_embed()`를 호출하면 다음 경로를 지난다.

```text
사용자 코드
  │
  │ nanoembed_embed(ctx, text, out)
  ▼
include/nanoembed/nanoembed.h
  공개 함수, 설정 구조체, 상태 코드
  │
  ▼
src/api/c_api.cpp
  포인터 검사, C 핸들을 C++ 객체로 연결, 예외를 오류 코드로 변환
  │
  ▼
src/embedder.cpp
  토큰화, 그래프 버퍼 준비, 계산 실행, 결과 복사
  ├─ src/tokenizer/*  문자열을 토큰 ID로 변환
  ├─ src/arch/*       GGUF 모델 구조에 맞는 인코더 그래프 선택
  ├─ src/forward/*    풀링과 BERT 세부 연산 그래프 구성
  └─ ggml             텐서 데이터 할당과 CPU 계산
  │
  ▼
호출자가 제공한 float[384] 버퍼
```

`c_api.cpp`는 C와 C++의 경계다. 실제 모델 계산은 `Embedder`와 그 아래 구성 요소가 맡는다.

## 모델을 로드할 때와 문장을 실행할 때

두 경로는 사용하는 데이터의 수명이 다르다.

### 모델 로드

```text
GGUF 파일
  → general.architecture 확인
  → BERT 메타데이터와 텐서 검사
  → 가중치 데이터 로드
  → tokenizer.ggml.model 확인
  → WordPiece 토크나이저 구성
  → 가중치 이름을 BERT 구조에 연결
```

이 데이터는 `nanoembed_model`이 소유한다. 모델 핸들이 살아 있는 동안 유지되며 여러 컨텍스트가 읽기 전용으로 공유할 수 있다.

### 컨텍스트 생성과 추론

```text
컨텍스트 설정
  → 최대 입력 길이에 필요한 그래프 버퍼 예약
  → 텍스트를 토큰 ID로 변환
  → 계산 그래프 구성
  → 예약한 버퍼에 입력과 중간 결과 배치
  → CPU에서 계산
  → 최종 임베딩을 호출자 버퍼로 복사
```

CPU 백엔드와 중간 결과 버퍼는 `nanoembed_context`가 소유한다. 같은 모델로 컨텍스트를 여러 개 만들면 각 컨텍스트가 별도 실행 버퍼를 가지므로 동시에 사용할 수 있다.

한 컨텍스트의 버퍼는 호출 중에 계속 바뀐다. 따라서 같은 컨텍스트를 두 스레드가 동시에 사용하면 안 된다.

## 현재 계층 구조

의존성은 위에서 아래로 흐른다. 아래 계층은 공개 C API를 알 필요가 없다.

```text
┌─────────────────────────────────────────────────────────┐
│ 공개 API       include/nanoembed/nanoembed.h            │
├─────────────────────────────────────────────────────────┤
│ C API 경계     src/api/c_api.cpp                        │
├─────────────────────────────────────────────────────────┤
│ 실행 조정      src/embedder.{h,cpp}                     │
├─────────────────────────────────────────────────────────┤
│ 모델 구조      src/arch/*                               │
│ 토크나이저     src/tokenizer/*                          │
├─────────────────────────────────────────────────────────┤
│ 공통 그래프    src/forward/*                            │
├─────────────────────────────────────────────────────────┤
│ 실행 기반      third_party/ggml                         │
└─────────────────────────────────────────────────────────┘
```

`src/arch/`는 모델마다 달라지는 인코더 계산과 가중치 구성을 감춘다. 현재는 BERT만 구현되어 있다. `src/tokenizer/`도 별도 인터페이스를 사용하며 현재는 WordPiece만 구현되어 있다.

모델 구조와 토크나이저를 분리한 이유는 둘이 독립적인 GGUF 메타데이터이기 때문이다. `general.architecture`는 인코더 계산을 선택하고, `tokenizer.ggml.model`은 문자열을 토큰 ID로 바꾸는 방법을 선택한다.

## 현재 디렉터리 구조

```text
NanoEmbed/
├── CMakeLists.txt
├── README.md
├── PLAN.md
├── include/nanoembed/
│   └── nanoembed.h                 공개 C API
├── src/
│   ├── api/c_api.cpp               C API 경계
│   ├── embedder.{h,cpp}            모델과 실행 흐름 조정
│   ├── gguf_scanner.{h,cpp}        BERT GGUF 검증
│   ├── arch/
│   │   ├── model_arch.h            모델 구조 공통 인터페이스
│   │   ├── registry.cpp            general.architecture 선택
│   │   └── bert_arch.{h,cpp}       BERT 가중치와 인코더 그래프
│   ├── tokenizer/
│   │   ├── tokenizer.h             토크나이저 공통 인터페이스
│   │   ├── registry.cpp            tokenizer.ggml.model 선택
│   │   └── wordpiece.{h,cpp}       BERT WordPiece 구현
│   └── forward/
│       ├── embed_layer.{h,cpp}      입력 임베딩
│       ├── attention.{h,cpp}        멀티헤드 셀프 어텐션
│       ├── ffn.{h,cpp}              피드포워드 네트워크
│       ├── encoder_block.{h,cpp}    어텐션과 FFN 연결
│       └── pool.{h,cpp}             풀링과 L2 정규화
├── tools/
│   ├── nanoembed-cli/              공개 API 사용 예
│   ├── nanoembed-inspect/          GGUF와 그래프 버퍼 검사
│   └── nanoembed-bench/            Linux 성능 측정 도구
├── bench/
│   ├── scenarios.yaml              측정 시나리오
│   ├── runner.py                   여러 시나리오 실행
│   ├── compare.py                  기준 측정값과 비교
│   └── baseline/                   커밋된 기준 측정값
├── tests/
│   ├── abi/                        공개 C ABI 테스트
│   ├── unit/                       스캐너·토크나이저·그래프 테스트
│   ├── integration/                최종 결과와 길이 경계 테스트
│   ├── fixtures/                   테스트 기준 데이터
│   └── corpus/                     평가 문장
└── third_party/ggml/               ggml 서브모듈
```

## 공개 헤더와 내부 구현의 경계

일반 사용자는 [`include/nanoembed/nanoembed.h`](../../../include/nanoembed/nanoembed.h)만 포함한다. `src/`의 C++ 클래스는 구현 세부 사항이다.

CMake는 `include/`를 `PUBLIC`, `src/`를 `PRIVATE` include 경로로 설정한다. 따라서 `nanoembed-cli`처럼 외부 사용자와 같은 조건의 실행 파일은 `Embedder`나 ggml 내부 타입에 직접 의존할 수 없다.

`nanoembed-inspect`와 일부 단위 테스트는 예외다. 이 도구들은 GGUF 구조와 내부 그래프를 검사해야 하므로 CMake에서 `src/` 접근을 명시적으로 허용한다.

이 경계에는 두 가지 목적이 있다.

1. 내부 구조를 바꿔도 공개 C ABI를 유지할 수 있다.
2. 공개 API만 사용하는 테스트가 실제 사용자와 같은 조건에서 빌드된다.

## 빌드 결과물

| 결과물 | 역할 | 기본 위치 |
|---|---|---|
| `nanoembed_core` | 정적 라이브러리 | `build/lib/` |
| `nanoembed-cli` | 가장 작은 공개 API 예제 | `build/bin/` |
| `nanoembed-inspect` | GGUF와 활성값 버퍼 검사 | `build/bin/` |
| `nanoembed-bench` | Linux 전용 성능 측정 | `build/bin/` |
| 테스트 실행 파일 | ABI, 단위, 통합 테스트 | `build/bin/` |

`nanoembed-bench`는 `/proc` 기능이 있는 Linux에서만 만들어진다. macOS에서 이 대상이 없는 것은 빌드 실패가 아니라 의도된 동작이다.

## 코드를 읽는 순서

1. 공개 핸들과 설정을 [`nanoembed.h`](../../../include/nanoembed/nanoembed.h)에서 본다.
2. [`nanoembed-cli`](../../../tools/nanoembed-cli/main.cpp)에서 모델과 컨텍스트 사용 순서를 본다.
3. [`c_api.cpp`](../../../src/api/c_api.cpp)에서 C 핸들이 C++ 객체로 연결되는 과정을 본다.
4. [`embedder.cpp`](../../../src/embedder.cpp)에서 로드와 추론 흐름을 구분한다.
5. [`registry.cpp`](../../../src/arch/registry.cpp)와 [`bert_arch.cpp`](../../../src/arch/bert_arch.cpp)에서 모델 구조 선택을 본다.
6. [`src/forward/`](../../../src/forward/)에서 BERT 계산을 구성하는 작은 그래프 함수를 읽는다.

4단계부터는 ggml의 텐서와 계산 그래프 개념이 필요하다. 다음 장에서 이 개념을 먼저 설명한다.

## 이 장의 핵심 정리

- 공개 C API는 구현 클래스와 ggml 타입을 감춘다.
- 모델은 가중치와 토크나이저를 소유하고 여러 컨텍스트가 공유한다.
- 컨텍스트는 변경 가능한 실행 버퍼를 소유하므로 동시에 사용하면 안 된다.
- 모델 구조와 토크나이저는 GGUF의 서로 다른 메타데이터로 선택한다.
- 현재 모델 가중치는 모두 메모리에 있으며, 레이어 스트리밍은 아직 구현 전이다.

[← 00. 가이드 목차](00-index.md) | [02. GGML과 GGUF 기초 →](02-ggml-primer.md)
