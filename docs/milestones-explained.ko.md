# NanoEmbed 마일스톤 해설서

이 문서는 각 마일스톤을 단순한 체크리스트가 아니라 다음 순서로 설명한다.

> 무엇인가 → 왜 필요한가 → 목표 → 가능한 방법 → 선택한 방식과 이유 → 실제 구현 → 현재 상태

일정과 수치 기준의 원본은 [`PLAN.md`](../PLAN.md)다. 여기서는 처음 보는 사람도 설계와 현재 상태를 이해할 수 있도록 배경을 풀어 쓴다. “구현”은 현재 코드에 있는 동작이고, “계획”은 아직 코드가 없는 설계안이다.

## 1. 프로젝트의 최종 목표

일반적인 임베딩 런타임은 모델 가중치를 메모리에 모두 올린 뒤 추론한다. 모델이 1 GB라면 가중치만으로도 그에 가까운 RAM이 필요하다. NanoEmbed는 이 관계를 끊으려 한다.

최종적으로 만들려는 실행 방식은 다음과 같다.

1. 모델은 GGUF 파일로 디스크에 둔다.
2. 토큰 임베딩처럼 항상 필요한 작은 부분만 계속 메모리에 둔다.
3. Transformer 레이어는 한 번에 하나만 읽는다.
4. 계산이 끝난 레이어의 가중치 페이지는 버린다.
5. 여러 문장은 같은 레이어가 메모리에 있는 동안 함께 처리한다.
6. 큰 배치에서 활성값이 병목이 되면 활성값도 압축한다.

따라서 최종 메모리 사용량을 다음 수준으로 제한하는 것이 목표다.

```text
상주 임베딩 테이블
+ 현재 레이어 하나의 가중치
+ 입력들의 활성값
+ 작은 런타임 오버헤드
```

이 구조를 한 번에 만들면 결과가 틀렸을 때 모델 수학, 파일 I/O, 메모리 수명 중 원인을 분리하기 어렵다. 그래서 먼저 정확한 기준 구현을 만든 뒤, 결과를 유지하면서 메모리 구조를 단계적으로 바꾼다.

## 2. 전체 마일스톤과 현재 위치

| 마일스톤 | 한 문장 목표 | 상태 |
|---|---|---|
| M1 | 빌드하고 호출할 수 있는 프로젝트와 C API 형태를 만든다 | 완료 |
| M2 | GGUF가 올바른 BERT인지 로딩 전에 검증한다 | 완료 |
| M3 | 메모리 제한을 무시한 정확한 BERT 임베더와 기준값을 만든다 | 완료 |
| M3.5 | 고정 256 MiB 계산 버퍼를 제거한다 | 완료 |
| M3.6 | 모델을 교체할 수 있게 만들고 두 번째 모델을 추가한다 | 완료 (벤치 baseline 제외) |
| M4 | 한 번에 한 레이어만 메모리에 두는 스트리밍을 만든다 | 완료 |
| M5 | 한 번 읽은 레이어를 여러 입력이 공유하는 실제 배치를 만든다 | 완료 (Harrier, BERT 패킹 제외) |
| M6 | 배치 활성값을 int8/int4로 압축한다 | 미구현·후순위 |
| M7 | 설치 가능한 라이브러리와 C++ 래퍼를 제공한다 | 미구현 |
| M8 | Node.js에서 비동기로 호출하게 한다 | 미구현 |

현재 위치는 **Harrier의 M5 실제 배치, 패딩 제거와 최종 파티션 판정을 끝낸 시점**이다.
두 번째 모델은 원래 후보였던 EuroBERT 대신 `microsoft/harrier-oss-v1-270m`(GGUF 태그
`gemma3`)으로 바꿨다. 이유는 7장에서 설명한다. Linux 레이어 스트리밍과
eager/streaming 실제 batch가 모두 있다. 다음 과제는 M5에서 확인한 그래프 경계
활성값과 `SlotStore` 메모리를 계측하고 줄이는 것이다.

---

## 3. M1 — 빌드 골격, ggml, 공개 C ABI, GGUF 검사

### 3.1 빌드 골격

**무엇인가.** 소스 파일을 어떤 언어 표준과 옵션으로 컴파일하고, 어떤 라이브러리를 연결하며, 실행 파일과 테스트를 어떻게 만드는지 정의하는 기본 구조다.

**왜 필요한가와 목표.** 사람마다 수동 명령이 다르면 결과를 재현하기 어렵다. macOS와 Linux에서 같은 명령으로 core library, 검사 도구, CLI와 테스트를 만들 수 있어야 한다.

**가능한 방법.** Makefile을 직접 쓰거나 CMake, Meson, Bazel 등을 쓸 수 있다. ggml을 시스템 패키지로 따로 설치하게 할 수도 있고 같은 빌드 트리에 포함할 수도 있다.

**선택과 이유.** C++ 생태계와 ggml이 이미 지원하며 M7의 설치 패키지로 이어지기 쉬운 CMake 3.18+를 선택했다. 별도 시스템보다 ggml의 기존 타깃을 그대로 재사용하기 쉽다.

**구현.** [`CMakeLists.txt`](../CMakeLists.txt)는 C++17/C11을 요구한다. NanoEmbed 코드에만 `-Wall -Wextra -Wpedantic`을 적용하고 서드파티 ggml에는 강요하지 않는다. `nanoembed_core`는 정적 라이브러리이며 공개 헤더는 `include/`, 내부 헤더는 `src/`로 분리한다. 외부 사용자가 우연히 내부 구현에 의존하는 것을 막기 위한 경계다.

**상태.** 완료됐다. shared library와 설치 규칙은 M7 범위다.

### 3.2 ggml 벤더링

**무엇인가.** ggml은 텐서 타입, 계산 그래프, CPU/Metal backend와 행렬 연산 커널을 제공한다. 벤더링은 특정 소스를 `third_party/ggml`에 고정해 함께 빌드한다는 뜻이다.

**왜 필요한가와 목표.** NanoEmbed의 차별점은 행렬 곱을 새로 만드는 것이 아니라 가중치 수명과 실행 순서를 바꾸는 데 있다. 검증된 연산을 재사용하고, ggml 버전 차이로 정확도와 메모리 결과가 달라지지 않게 해야 한다.

**가능한 방법.** 커널을 직접 구현하거나, 시스템 ggml을 찾거나, 빌드 때 다운로드하거나, git submodule로 버전을 고정할 수 있다.

**선택과 이유.** git submodule과 CMake `add_subdirectory`를 선택했다. 직접 커널을 만들면 범위가 지나치게 커지고, 시스템 ggml은 머신별 버전 차이로 재현성이 낮다. ggml 자체 테스트와 예제는 꺼서 NanoEmbed 빌드를 작게 유지한다.

**상태.** 완료됐다. 현재 F16/F32와 Harrier Q8_0 인메모리 경로는 ggml 커널로
자동 검증한다. Q4 계열은 수치를 기록했으며, 실제 스트리밍 제품 경로는 M4에서
정밀도별로 다시 검증한다.

### 3.3 공개 C ABI 동결

**무엇인가.** API는 소스 수준의 호출 계약이고 ABI는 함수 이름, 인자 배치와 구조체 크기 같은 바이너리 계약이다. 공개 C ABI를 동결한다는 것은 내부 구현이 바뀌어도 외부 프로그램이 같은 함수 형태로 호출하게 한다는 뜻이다. 계약은 [`include/nanoembed/nanoembed.h`](../include/nanoembed/nanoembed.h)에 있다.

**왜 필요한가와 목표.** Node.js, Swift, Rust, Python 등은 C ABI를 쉽게 감쌀 수 있다. C++ 클래스를 그대로 공개하면 컴파일러와 표준 라이브러리에 ABI가 묶인다. M3, M4, M5에서 실행 방식이 바뀌어도 사용자 코드는 바뀌지 않아야 한다.

모델과 실행 상태를 다음처럼 나눈다.

- `nanoembed_model`: 읽기 전용 모델과 가중치
- `nanoembed_context`: 스레드, 최대 길이, 계산 버퍼 같은 호출 상태
- 출력 버퍼: 호출자 소유
- 오류: 음수 상태 코드와 `nanoembed_last_error()`

**가능한 방법.** C++ 클래스를 공개하거나, C ABI를 핵심 계약으로 삼을 수 있다. 출력도 라이브러리가 새로 할당해 반환하거나 호출자가 버퍼를 전달하게 할 수 있다.

**선택과 이유.** 불투명 포인터 C ABI와 caller-owned 출력 버퍼를 선택했다. 실제 구조를 헤더에서 숨겨 내부 필드를 바꾸기 쉽고, 언어 경계에서 메모리 소유권이 명확하며, 반복 호출 때 출력 메모리를 재사용할 수 있다. C++ 예외는 공개 함수 안에서 잡아 C 경계를 넘지 않게 한다.

**구현.** 모델 로드/해제, 차원 조회, context 생성/해제, 단일/배치 임베딩, 오류 조회 함수를 제공한다. M1에는 심볼과 stub만 있었고 M3에서 같은 함수에 실제 동작을 채웠다. 순수 C인 [`tests/abi/abi_link_test.c`](../tests/abi/abi_link_test.c)가 모든 공개 심볼에 링크되는지 검사한다.

**주의.** 현재 “동결”은 함수 시그니처를 함부로 바꾸지 않는다는 뜻이다. 값으로 전달하는 `nanoembed_context_params`의 크기를 늘리면 ABI가 깨질 수 있으므로, 향후 필드는 버전이 붙은 새 함수나 크기 필드가 있는 구조체로 확장하는 편이 안전하다.

**상태.** API 모양과 링크 테스트는 완료됐다. shared library의 SOVERSION 보장은 M7에서 한다.

### 3.4 GGUF smoke test와 검사 도구

**무엇인가.** smoke test는 모든 기능보다 파일을 열고 기본 구조를 읽는 최소 확인이다. `nanoembed-inspect`는 GGUF 버전, metadata, tensor와 모델 구조를 사람이 보게 한다.

**왜 필요한가와 목표.** 모델 변환기마다 key와 tensor 이름이 다를 수 있다. 추론부터 만들면 오류가 파일 문제인지 수학 문제인지 구분하기 어렵다. 먼저 실제 파일을 관찰할 도구가 필요하다.

**가능한 방법과 선택.** 별도 Python 파서를 쓸 수도 있지만 제품과 다른 파서가 다르게 해석할 위험이 있다. 실제 런타임과 같은 ggml GGUF 파서를 호출하는 작은 C++ 도구를 선택했다.

**구현.** [`tools/nanoembed-inspect/main.cpp`](../tools/nanoembed-inspect/main.cpp)는 기본 실행에서 헤더와 metadata KV를 출력한다. `--tensors`는 전체 tensor 표를, `--graph`는 실제 모델을 열어 활성값 버퍼 예약량을 보여 준다. `--graph`는 weight load와 graph 구성이 필요하므로 opt-in이다.

**상태.** 완료됐다. M2 이후 BERT manifest도 보여 주고, M3.6 이후에는 인식하지만 지원하지 않는 아키텍처도 구체적으로 알린다.

### 3.5 CI와 라이선스

**무엇이며 왜 필요한가.** CI는 변경마다 깨끗한 환경에서 빌드와 테스트를 반복해 특정 개발 머신에서만 동작하는 상태를 막는다. 라이선스는 코드와 dependency를 사용할 조건을 명시한다.

**구현과 상태.** GitHub Actions로 macOS/Linux를 확인한다. 모델이 없어 정확도 테스트가 조용히 skip되지 않게 CI에서는 `NANOEMBED_REQUIRE_MODEL`로 모델 존재를 강제할 수 있다. [`LICENSE`](../LICENSE)가 있으며 배포물의 ggml 고지는 M7에서 다시 검토해야 한다.

---

## 4. M2 — GGUF 스캐너와 BERT manifest

### 4.1 `no_alloc` 메타데이터 스캔

**무엇인가.** GGUF의 설정 KV와 tensor 이름, shape, dtype, 파일 offset만 읽고 실제 weight 데이터는 올리지 않는 단계다.

**왜 필요한가와 목표.** 잘못된 모델은 첫 추론 중 행렬 곱에서 죽는 대신 로딩 때 명확히 거부돼야 한다. 수 GB 모델도 검증만 하려고 수 GB를 RAM에 읽어서는 안 된다.

**가능한 방법.** 전체 로드 후 검사, 직접 GGUF 파싱, ggml의 metadata-only 모드가 있다.

**선택과 이유.** `gguf_init_from_file(..., no_alloc=true)`를 선택했다. 제품과 같은 파서를 쓰면서 weight allocation만 피한다.

**구현.** [`src/gguf_scanner.cpp`](../src/gguf_scanner.cpp)는 필수 KV의 존재와 타입을 검사한다. `TensorRef`에 GGUF index, ggml dtype, shape, byte 크기와 data offset을 기록한다. `ScanResult`가 GGUF와 ggml metadata context를 RAII로 소유해 중간 예외에도 해제한다.

**상태.** BERT에 대해 완료됐다.

### 4.2 평탄한 tensor 목록을 `ModelManifest`로 변환

**무엇인가.** GGUF의 `blk.0.attn_q.weight` 같은 이름을 “0번 레이어 Q weight”처럼 의미가 붙은 슬롯으로 번역한 결과다.

**왜 필요한가와 목표.** forward 곳곳에서 문자열로 tensor를 찾으면 오타와 누락이 늦게 발견된다. 한 번 검증한 뒤 타입 있는 구조를 넘기면 이후 코드는 올바른 모델이라는 전제를 쓸 수 있다.

**가능한 방법.** 매번 문자열 조회, enum만 사용, 아키텍처별 구조체에 모든 참조 저장이 있다.

**선택과 이유.** v0 지원 모델이 BERT 하나였으므로 명시적인 `ModelManifest`와 레이어별 `LayerSlot`을 선택했다. 존재하지 않는 범용성을 위한 거대한 추상화보다 검증하기 쉽다.

**구현.** [`src/gguf_scanner.h`](../src/gguf_scanner.h)는 hyperparameter, token/position/type embedding, embedding LayerNorm, 각 레이어의 Q/K/V/O, 두 LayerNorm, FFN up/down과 선택적 pooler를 보관한다.

**상태와 변화.** 완료됐다. M3.6에서는 이를 범용 manifest로 억지 확장하지 않고 BERT 내부 세부사항으로 유지했다. EuroBERT와 겹치는 tensor가 거의 없기 때문이다.

### 4.3 fail-fast shape 검증

**무엇인가.** 필수 metadata, tensor 존재와 shape가 다르면 모델 생성 즉시 실패시키는 정책이다.

**왜 필요한가와 목표.** hidden 384 모델에 `[768,768]` weight가 들어 있으면 계산을 시작해서는 안 된다. 로드 단계의 구체적인 오류가 추론 중의 일반 internal error보다 진단하기 쉽다.

**선택과 구현.** ggml 연산 실패까지 기다리지 않고 전부 사전 검증한다. hidden이 head 수로 나누어지는지, Q/K/V/O가 `[H,H]`, FFN up/down이 `[H,F]`/`[F,H]`, bias와 norm이 1차원인지 확인하고 `ScanError`로 보고한다.

**상태.** 완료됐고 scanner 단위 테스트가 있다.

---

## 5. M3 — 정확한 인메모리 기준 임베더

M3의 목적은 최종 메모리 구조가 아니라 **수학적으로 맞는 기준 구현과 oracle**이다. 스트리밍과 배치를 먼저 넣으면 오류 원인이 수학인지 I/O인지 분리하기 어렵기 때문에 모든 weight를 RAM에 올리는 단순 경로부터 완성했다.

### 5.1 WordPiece 토크나이저

**무엇인가.** 문자열을 모델 vocab의 정수 ID로 바꾼다. BERT는 단어와 문장부호를 나눈 뒤 longest-prefix WordPiece로 subword를 찾고 `[CLS]`, `[SEP]`을 붙인다.

**왜 필요한가와 목표.** forward가 맞아도 token ID 하나가 다르면 임베딩이 달라진다. Hugging Face tokenizer와 ID가 완전히 같아야 한다.

**가능한 방법.** Hugging Face를 런타임에서 호출하거나, 외부 tokenizer library를 포함하거나, 필요한 WordPiece만 C++로 구현할 수 있다.

**선택과 이유.** 엣지용 C++ 단일 라이브러리와 BERT 한 종류라는 범위에 맞춰 인트리 구현을 선택했다. Python/Rust runtime dependency가 없고 규모가 관리 가능하다.

**구현.** [`src/tokenizer/wordpiece.cpp`](../src/tokenizer/wordpiece.cpp)는 GGUF vocab과 특수 token ID를 읽고, llama.cpp 변환기의 `▁piece` 표기를 Hugging Face식 `piece`/`##piece`로 바꾼다. ASCII 공백/control/punctuation과 소문자화를 처리한 뒤 각 단어에서 가장 긴 vocab prefix를 탐욕적으로 찾는다. 실패하거나 100자를 넘는 단어는 `[UNK]`, 본문은 길이에 맞춰 truncate하고 양끝에 `[CLS]`/`[SEP]`을 붙인다.

**검증과 상태.** HF가 생성한 [`bge-small-eval.tsv`](../tests/fixtures/tokenizer/bge-small-eval.tsv)와 ID 완전 일치를 검사한다. BERT 범위는 완료됐다. 비ASCII 전체 정규화를 재현하는 범용 BasicTokenizer는 아니며 byte-level BPE는 M3.6 범위다.

### 5.2 상태 없는 forward graph builder

**무엇인가.** 실제 값을 즉시 계산하는 대신 ggml 연산 노드로 embedding, attention, FFN, pooling graph를 만든다.

**왜 필요한가와 목표.** HF BERT와 같은 출력을 내면서 M4에서는 수학을 바꾸지 않고 weight 수명만 바꿔야 한다. 함수가 파일이나 전역 상태를 소유하면 레이어 하나만 떼어 스트리밍하기 어렵다.

**가능한 방법.** 전체를 거대한 클래스 하나로 만들거나, I/O와 계산을 레이어 객체가 같이 맡거나, 입력과 weight만 받는 작은 stateless builder로 나눌 수 있다.

**선택과 이유.** 마지막 방식을 선택했다. M3은 전체 weight를 넘기고 M4는 현재 레이어 weight만 넘기면서 같은 계산 함수를 재사용할 수 있다.

**구현.** 텐서 차원은 `[H,S,B]`로 통일했다.

- [`embed_layer.cpp`](../src/forward/embed_layer.cpp): token/position/type lookup, 합산, LayerNorm
- [`attention.cpp`](../src/forward/attention.cpp): Q/K/V, multi-head reshape, scaled softmax, output projection, residual, post-LN
- [`ffn.cpp`](../src/forward/ffn.cpp): up, HF와 같은 erf GeLU, down, residual, post-LN
- [`encoder_block.cpp`](../src/forward/encoder_block.cpp): attention과 FFN 연결
- [`pool.cpp`](../src/forward/pool.cpp): mean/CLS/LAST와 L2 normalization

**상태와 제한.** 단일 입력은 mask 없는 `B=1` 경로를 유지한다. M5 batch는 optional
padding attention mask와 Mean/LAST용 mask/index를 전달한다.

### 5.3 `Embedder` façade와 소유권

**무엇인가.** tokenizer, architecture, graph 구성과 backend 실행을 단순한 인터페이스 뒤에 묶는 객체다.

**왜 필요한가와 목표.** 공개 API가 ggml 세부사항을 몰라도 되게 하고, 공유 가능한 읽기 전용 model과 context별 mutable 계산 상태를 분리한다.

**가능한 방법.** 전역 singleton, 모델과 scratch를 한 객체에 묶기, 모델은 weight를 context는 scratch를 소유하기가 있다.

**선택과 이유.** `nanoembed_model`이 `Embedder`와 weight를, `nanoembed_context`가 설정과 `ComputeScratch`를 소유한다. 서로 다른 context가 같은 읽기 전용 모델을 공유하면서 각자 thread-unsafe backend/allocator를 갖게 해 동시성 계약을 지킨다.

**구현.** [`src/embedder.cpp`](../src/embedder.cpp)는 architecture를 고른 뒤 GGUF를 `no_alloc=false`로 다시 열어 현재는 모든 weight를 RAM에 올린다. tokenizer와 weight를 연결한 후 `embed()`에서 tokenize → input graph 생성 → architecture forward → pooling/normalization → graph allocation → input copy → CPU compute → caller buffer copy를 수행한다.

**상태와 제한.** 단일 입력은 완료됐다. 모든 weight가 상주하므로 최종 메모리 목표는 M4 전까지 달성되지 않는다.

### 5.4 C API 구현과 오류 변환

**무엇인가.** M1 stub을 실제 `Embedder` 호출로 연결하고 C++ 오류를 C 계약으로 바꾸는 작업이다.

**가능한 방법.** 실패 시 종료, 오류 무시, 매 함수에 error buffer 전달, 상태 코드와 thread-local 오류 문자열이 있다.

**선택과 이유.** 마지막 방식을 선택했다. 함수가 작고 서로 다른 스레드의 오류가 덮어써지지 않는다.

**구현.** [`src/api/c_api.cpp`](../src/api/c_api.cpp)는 null/범위를 먼저 검사하고 예외를
잡아 음수 코드와 최대 512자 thread-local 메시지로 변환한다. free는 null-safe다.
`nanoembed_embed_batch`는 eager/Linux streaming의 실제 batch 경로를 호출하며
invalid argument는 출력을 보존하고 실행 후 실패는 전체 출력을 NaN으로 만든다.

**상태.** 단일과 실제 batch가 작동한다. 공개 ABI layout은 M1 때와 같다.

### 5.5 CLI

**무엇이며 왜 필요한가.** stdin 한 줄을 임베딩해 CSV float를 출력하는 공개 C API 사용 예제이자 end-to-end smoke test다.

**선택과 구현.** 큰 데모 앱 대신 pipe와 자동화에 쉬운 최소 CLI를 만들었다. [`tools/nanoembed-cli/main.cpp`](../tools/nanoembed-cli/main.cpp)는 모델, CLS, normalization, thread 수와 최대 길이 옵션을 받는다. 내부 전용 API가 아니라 공개 C API만 호출한다.

**상태.** 완료됐다. M4의 `--streaming` 옵션은 아직 없다.

### 5.6 정확도 oracle과 테스트 피라미드

**무엇인가.** oracle은 정답으로 간주할 결과다. tokenizer, 레이어 중간값, 최종 embedding을 서로 다른 깊이에서 비교한다.

**왜 필요한가와 목표.** 최종 벡터만 틀렸다고 알면 처음 오차가 난 위치를 찾기 어렵다. 단계별 fixture로 문제 범위를 줄인다.

**가능한 방법.** 눈으로 확인, crash만 검사, 실행 때마다 upstream 모델 호출, 고정 fixture 비교가 있다.

**선택과 이유.** HF/sentence-transformers 결과를 고정 fixture로 저장했다. CI가 Python 모델을 매번 받을 필요가 없고 upstream 버전 때문에 정답이 조용히 바뀌지 않는다.

**구현.** tokenizer fixture는 ID 완전 일치, activation fixture는 embedding과 12개 encoder block의 출력, golden fixture는 최종 벡터 cosine을 검사한다. 도구는 `dump_tokenizer_fixture.py`, `dump_hf_activations.py`, `dump_golden.py`다. M3 기준은 sentence-transformers 대비 최소 cosine 0.9999, 평균 0.99999 이상이다.

**상태.** BERT와 Harrier의 tokenizer·activation·golden fixture가 모두 CI에 연결돼
있다. Harrier Q8_0도 같은 F32 golden에 대해 별도의 허용 범위로 자동 검사한다.
스트리밍 경로의 양자화 oracle은 M4에서 다시 검증해야 한다.

### 5.7 벤치마크와 baseline

**무엇인가.** latency뿐 아니라 RSS/PSS/USS, page fault, CPU, disk read와 throughput을 측정하고 마일스톤별 JSON을 비교한다.

**왜 필요한가와 목표.** 프로젝트의 핵심 주장은 메모리 절감이다. 정확도만으로 완료를 판단할 수 없고, 메모리를 줄이면서 생기는 I/O/latency 비용도 기록해야 한다.

**가능한 방법.** process 내부에서 한 번 측정, 외부 `time`, 같은 process에서 시나리오 연속 실행, 깨끗한 worker를 부모가 관찰하는 방식이 있다.

**선택과 이유.** Linux `/proc` 기반 `fork + exec` worker 격리를 선택했다. 단순 fork는 부모의 dirty page를 자식 RSS에 물려주지만 exec는 주소 공간을 교체한다. RSS는 값싼 `statm`을 자주 읽고 page table을 걷는 `smaps_rollup`의 PSS/USS는 드물게 읽는다. peak RSS는 커널 `VmHWM`을 사용하며 측정 창 전에 `clear_refs`로 기준을 재설정한다.

**구현.** [`tools/nanoembed-bench/`](../tools/nanoembed-bench)가 측정하고, [`scenarios.yaml`](../bench/scenarios.yaml)이 workload를, [`runner.py`](../bench/runner.py)가 JSON 집계를, [`compare.py`](../bench/compare.py)가 환경 fingerprint와 회귀를 비교한다. kernel, CPU, CPU 수와 page size가 다르면 strict 비교를 거부한다.

**상태와 제한.** [`M3.json`](../bench/baseline/M3.json)과 [`M3.5.json`](../bench/baseline/M3.5.json)이 있다. 정확한 `/proc` 동작에 의존해 bench는 Linux 전용이다.

---

## 6. M3.5 — 고정 256 MiB 그래프 아레나 제거

### 6.1 문제와 목표

M3 최초 구현은 매 `embed()`마다 256 MiB byte vector를 만들고 일반 ggml context를 열었다. 기본 context는 bump allocator라 중간 tensor의 수명이 끝나도 공간을 재사용하지 않는다. 단문은 약 12 MiB만 쓰는데 vector 초기화로 256 MiB 전체가 RSS에 잡혔고, 약 270 token부터 고정 arena가 부족해 assert로 종료됐다. 공개 최대 길이 512와 구현이 맞지 않았다.

목표는 수학과 결과는 M3 그대로 유지하면서 다음을 달성하는 것이다.

- tensor 수명이 겹치지 않으면 메모리를 재사용한다.
- context 생성 때 최대 길이를 감당할 수 있는지 확인한다.
- 호출마다 큰 buffer를 재할당·memset하지 않는다.
- 512 token과 장문/단문 교차 호출을 안전하게 처리한다.

### 6.2 가능한 방법과 선택

arena를 1 GiB로 키우기, 입력 길이로 크기를 수동 계산하기, 레이어마다 context를 만들기, graph liveness allocator를 쓰기가 있다. 크기 확대는 문제를 늦출 뿐이고, 수동 계산은 연산 변경 때 틀리기 쉽다. 실제 graph를 보고 동시에 살아 있는 tensor만 분리하는 `ggml_gallocr`를 선택했다.

### 6.3 구체적인 구현

현재 [`ComputeScratch`](../src/embedder.h)는 context별 CPU backend, gallocr, graph metadata buffer와 예약 길이를 소유한다. graph context는 `no_alloc=true`여서 tensor 구조체만 만들고 실제 data 위치는 gallocr가 정한다. 입력은 `ggml_set_input`, 결과는 `ggml_set_output`으로 표시한다.

context 생성 때 `max_seq_len` graph를 만들어 `ggml_gallocr_reserve`를 호출한다. 입력은 graph allocation 뒤 `ggml_backend_tensor_set`으로 넣는다. 그 전에는 data pointer가 없기 때문이다. M3.6 로컬 변경은 mutable scratch를 model에서 context로 옮겨 여러 context가 allocator/backend를 공유하지 않게 했다.

### 6.4 검증과 결과

[`seq_len_test.cpp`](../tests/integration/seq_len_test.cpp)는 512 token, 초과 truncate, 장문/단문 교차 실행의 결과 오염을 검사한다. 계획서의 동일 머신 결과에서 단문 peak RSS는 330.05 MiB에서 76.35 MiB로, p50은 240.05ms에서 41.97ms로 줄었다. 이는 수학 최적화가 아니라 호출마다 256 MiB를 초기화하던 비용을 없앤 결과다.

### 6.5 현재 상태와 주의점

마일스톤은 완료됐고 M3.5 baseline이 있다.

`reserve`가 context 설정이 아니라 기본값으로 예약하던 문제는 고쳤다.
[`reserve_invariant_test.cpp`](../tests/integration/reserve_invariant_test.cpp)가 두 모델에서
최대 길이 예약 뒤 첫 호출 전후 buffer 크기가 같은지 검사한다.

다만 **로컬 CLI smoke test에서 한 번 나왔던 gallocr 자동 재할당 진단은 여전히 원인
미상이다.** 위 수정이 그 원인이었을 것으로 추정했으나, 실측 결과 예약 buffer 크기는
pooling과 normalization 조합 전체에서 바이트 단위로 동일했다(bge-small 16521216,
harrier 15208448). 어텐션 활성값이 예약을 지배하고 pooling·normalization이 덧붙이는
연산은 그 안에 흡수되기 때문이다. 따라서 그 수정은 재할당 진단의 원인이 될 수 없고,
이 항목은 열린 채로 둔다. "모든 입력이 예약 buffer만 사용한다"는 강한 불변식을
주장하려면 진단을 재현해 원인을 찾아야 한다.

---

## 7. M3.6 — 모델 교체 구조와 Harrier 지원

M3.6은 원래 M1~M8 사이에 없던 선행 단계다. 작은 BERT만으로는 M4 스트리밍 절감
효과를 충분히 보여 주기 어려워 더 큰 두 번째 계열을 추가하면서 생겼다. 최초 후보였던
Jina v5 Nano 대신 라이선스와 후속 확장성을 검토해 `microsoft/harrier-oss-v1-270m`
(`gemma3`)을 최종 대상으로 선택했다.

### 7.1 아키텍처 교체 경계

**무엇인가.** 모델마다 metadata key, 필수 tensor, position encoding, normalization과 FFN이 다르다. 이를 `ModelArch` 뒤에 숨겨 공통 실행기가 구체적인 모델 종류를 몰라도 되게 한다.

**왜 필요한가와 목표.** BERT는 learned position embedding, LayerNorm, GELU와 bias
projection을 쓰지만 Harrier는 RoPE, RMSNorm, GeGLU, GQA와 causal attention을 쓴다.
BERT manifest에 optional 필드를 계속 더하면 거대한 조건문이 된다. 새 모델을 추가할
때 기존 BERT 코드를 수정하지 않는 것이 목표다.

**가능한 방법.** forward 각 단계에 `if (arch...)`를 넣거나, 모든 모델을 담는 거대 manifest를 만들거나, metadata/weight binding/graph build 전체를 architecture 객체로 분리할 수 있다.

**선택과 이유.** 마지막 방식을 택했다. 공통점이 적은 모델을 세부 연산 수준에서 억지로 추상화하지 않고 graph 전체라는 높은 경계에서 나눈다. 두 번째 구현이 실제로 생길 때 추상화한다는 원칙에도 맞는다.

**구현.** [`src/arch/model_arch.h`](../src/arch/model_arch.h)는 공통 parameter, 필요한
graph input, 기본 pooling, weight binding과 encoder graph build 계약을 정의한다.
[`src/arch/registry.cpp`](../src/arch/registry.cpp)는 `general.architecture`를 읽어
`bert`와 `gemma3`를 각각의 구현으로 연결한다. `eurobert` 태그는 인식하지만 명시적인
미구현 오류를 낸다.

**상태.** 교체 구조와 BERT·Gemma 3 두 구현이 완료됐다. EuroBERT 구현체는 범위 밖이다.

### 7.2 tokenizer를 architecture와 독립적으로 교체

**무엇인가.** `general.architecture`와 `tokenizer.ggml.model`은 별개다. 모델 수학과 문자열 분할 방식이 항상 일대일로 묶이지 않는다.

**왜 필요한가와 목표.** architecture registry 안에서 tokenizer까지 고르면 같은 모델 구조가 다른 tokenizer를 쓰는 경우를 지원하기 어렵다. 두 축을 독립 조합하는 것이 목표다.

**가능한 방법과 선택.** 모델 클래스가 tokenizer를 직접 만들게 할 수도 있지만, GGUF의 독립 metadata를 그대로 반영해 별도 `Tokenizer` interface와 registry를 선택했다.

**구현.** [`src/tokenizer/tokenizer.h`](../src/tokenizer/tokenizer.h)는 `encode`, 최대 길이와
vocab 계약을 정의한다. [`registry.cpp`](../src/tokenizer/registry.cpp)는 `bert`를
WordPiece, `llama`를 SentencePiece 계열 BPE로 연결한다. `gpt2` byte-level BPE는
태그만 인식하고 미구현 오류를 낸다.

**상태.** Harrier에 필요한 BPE까지 완료됐다.

### 7.3 LAST pooling

**무엇이며 왜 필요한가.** token별 hidden state를 문장 벡터 하나로 줄여야 한다.
BERT는 CLS를, Harrier는 마지막 유효 token을 선택한다.

**가능한 방법과 선택.** 모델 graph 안에 pooling을 중복 구현하거나 공통 builder에 종류를 추가할 수 있다. 수학을 공유할 수 있으므로 공통 builder에 `Last`를 추가하고 모델이 기본 종류를 고르게 했다.

**구현.** [`build_last_pool`](../src/forward/pool.cpp)은 단일 입력에서는 마지막 S 위치를
선택한다. padded batch에서는 `[H,S*B]`로 flatten한 뒤 문장별 마지막 유효 index를
`get_rows`로 gather한다.

**상태.** 공개 C ABI의 `NANOEMBED_POOL_LAST`와 `NANOEMBED_POOL_MODEL_DEFAULT`까지
완료됐다. 기본값은 모델이 학습된 pooling이며 호출자가 Mean/CLS/LAST로 덮어쓸 수 있다.

### 7.4 긴 context의 예약 정책

**무엇인가.** Harrier는 최대 길이 32768을 선언한다. attention score 메모리는 길이
제곱에 비례하므로 모델 최대치를 무조건 예약하면 현실적으로 감당할 수 없는 공간이
필요할 수 있다.

**가능한 방법.** 모델 최대를 항상 예약, 첫 요청에서 동적 예약, API의 `max_seq_len` 기준으로 context 생성 때 예약할 수 있다.

**선택과 이유.** 세 번째 방식이다. 첫 요청 중간 OOM보다 context 생성에서 실패시키고, 사용자가 실제 필요한 최대 길이 비용을 명시하게 한다. 기본 cap은 512이며 모델 최대를 넘으면 clamp한다.

**구현과 상태.** `nanoembed_new_context`가 context별 `ComputeScratch`를 만들고 cap 기준으로
`Embedder::reserve`한다. tokenizer도 같은 effective limit을 쓴다. 두 지원 토크나이저가
앞뒤 특수 token을 붙이므로 공개 `max_seq_len`은 2 이상이어야 하며, 1은 context 생성
때 거부한다. 길이 2와 초과 truncate 회귀 테스트가 이 계약을 고정한다.

### 7.5 Hugging Face 모델 source

**무엇이며 왜 필요한가.** 대형 GGUF를 저장소에 commit하지 않고도 정확한 repo/filename을 bench에 기록한다. 모호한 모델 이름은 다른 quantization을 잘못 측정할 수 있다.

**가능한 방법과 선택.** bench가 자동 다운로드할 수도 있지만 네트워크와 upstream 변경이 측정에 섞인다. `hf:<repo>:<filename>`을 로컬 HF cache에서만 찾고, 없으면 다운로드 명령만 안내하도록 했다.

**구현과 상태.** [`bench/model_source.py`](../bench/model_source.py)는 명시적 Hugging Face
source를 해석할 수 있고, CI와 README는 Harrier F32/Q8_0의 정확한 저장소와 파일명을
고정한다. M3.6 시나리오는 BERT 단문·장문과 Harrier F32를 모두 선택한다. baseline이
없는 이유는 모델 미구현이 아니라 `/proc` 기반 벤치가 Linux 전용이기 때문이다.

### 7.6 Harrier forward

Harrier 경로는 전용 GGUF manifest와 `Gemma3ModelArch`로 구현했다. RoPE, RMSNorm,
QK-norm, query 4개가 하나의 KV head를 공유하는 GQA, causal mask, GeGLU와 분리된
`head_dim`을 ggml primitive로 조합한다. 스캐너가 관련 metadata의 자료형·범위와 모든
필수 tensor shape를 실행 전에 검증한다. 레이어별 activation fixture가 처음 오차가
나는 블록을 찾고 최종 golden이 전체 경로를 검사한다.

### 7.7 SentencePiece 계열 BPE

Harrier의 `llama` tokenizer는 GGUF vocab, merge 순위, byte fallback과 BOS/EOS template를
읽는 독립 구현이다. 특히 GGUF의 `add_eos_token=false`보다 변환기가 기록한
`suffix_token_id`를 우선한다. 마지막 token을 pooling하는 모델이라 EOS가 빠지면 완전히
다른 벡터가 되기 때문이다. HF fixture 132문장과 token ID가 완전히 일치한다.

남은 제한은 입력 문자열 안의 added-token literal(예: 문자 그대로의 `<eos>`)을 HF처럼
정규화 전에 분리하지 않는다는 점이다. 일반 텍스트와 모델이 자동으로 붙이는 특수
token에는 영향이 없고, 향후 범용 tokenizer 지원에서 별도로 다룬다.

### 7.8 M3.6 완료 조건과 결과

대상 모델을 `microsoft/harrier-oss-v1-270m`(`gemma3`)으로 바꿨다. EuroBERT를 쓰지
않은 이유는 두 가지다. jina 쪽은 **cc-by-nc-4.0(비상업)** 이라 fixture와 baseline이
저장소에 들어가면 되돌리기 어렵고, Harrier가 요구하는 메커니즘(GQA, causal 마스킹,
QK-norm, head_dim 분리)은 요즘 decoder-only 모델의 표준이라 다음 모델로 그대로
이어진다. `gemma3`가 llama.cpp의 1급 아키텍처라 대조할 참조 구현이 있다는 점도 컸다.

| 조건 | 결과 |
|---|---|
| 새 모델 forward | 완료 (RoPE, RMSNorm, GeGLU, GQA, QK-norm, causal) |
| 새 토크나이저 | 완료 (SentencePiece 계열 BPE) |
| 공개 LAST pooling | 완료. 기본값을 "모델이 학습된 풀링"으로 바꿨다 |
| HF tokenizer ID 완전 일치 | 132/132 |
| 레이어별 활성값 대조 | 상대오차 1e-6~2e-5, 임베딩 단계는 정확히 일치 |
| sentence-transformers golden | 132/132, 코사인 1.000000 |
| Q8_0 자동 회귀 | 완료. 문장별 0.9985, 평균 0.9995 이상 |
| context 동시성 | 완료. 공유 모델 + 서로 다른 context 2개 병렬 실행 |
| 예약 buffer 불변식 | 완료. 최대 길이 첫 실행 전후 크기 동일 |
| M3.6 벤치 시나리오 | BERT 단문·장문 + Harrier F32 선택 |
| `bench/baseline/M3.6.json` | **미완**. Linux 전용 도구라 해당 머신에서 별도 측정 |

양자화 손실은 F32 기준값과 따로 잰다. 구현이 맞는지와 양자화 오차가 얼마인지는
서로 다른 검사 항목이라 하나의 임계값으로 묶으면 원인 구분이 불가능해진다.
아래는 같은 sentence-transformers F32 기준값에 대해 132문장을 잰 결과다.
F32 경로가 그 기준값과 1.000000으로 일치하므로, 남은 차이는 전부 양자화 오차다.

| 파일 | 크기 | 최소 코사인 | 평균 코사인 |
|---|---:|---:|---:|
| F32 | 1088 MB | 1.000000 | 1.000000 |
| q8_0 | 301 MB | 0.999117 | 0.999754 |
| q5_k | 263 MB | 0.981933 | 0.995445 |
| q4_k | 251 MB | 0.946921 | 0.984729 |

두 가지를 읽어야 한다.

첫째, **평균은 최악의 경우를 가린다.** q4_k의 평균은 0.9847로 그럴듯하지만 최소는
0.9469다. 검색 품질은 평균이 아니라 순위가 뒤집히는 문장에서 결정되므로, 이런
지표는 평균만 보고 판단하면 안 된다.

둘째, **더 세게 양자화해도 파일이 별로 줄지 않는다.** q8_0에서 q4_k로 가면 품질은
확실히 나빠지는데 크기는 17%만 줄어든다. 토큰 임베딩 표가 세 변형 모두에서 q8_0로
유지되기 때문이다(178 MB 고정). 줄어드는 것은 블록뿐이고, 그건 파일의 37%다.
같은 이유로 M4의 메모리 문제도 양자화로는 풀리지 않는다.

이 단계에서 **추측했으면 틀렸을 것**이 셋 있었고, 전부 실물 파일과 참조 구현을
직접 열어 확인했다. 자세한 내용은 `src/arch/gemma3_arch.h`의 머리말에 적어 두었다.

1. Gemma의 RMSNorm은 `x * (1 + w)`인데, 변환기가 그 `+1`을 가중치에 이미 접어
   넣었다. 한 번 더 더하면 전 구간이 틀어진다.
2. 반대로 임베딩의 `sqrt(n_embed)` 스케일은 접혀 있지 않아 그래프가 적용해야 한다.
3. GGUF는 `add_eos_token=false`라고 적어놨지만 HuggingFace는 `<eos>`를 붙인다.
   마지막 토큰으로 풀링하는 모델에서 이걸 놓치면 아무 신호 없이 다른 벡터가 나온다.

---

## 8. M4 — 레이어 스트리밍과 양자화 weight

> M4부터는 현재 선택된 설계 계획이다. 아래 runtime 파일과 검증 결과는 아직 없다.

### 8.1 레이어 스트리밍

**무엇인가.** 전체 Transformer weight 대신 0번 레이어를 읽고 계산한 뒤 버리고, 다음 레이어를 읽는 실행 방식이다.

**왜 필요한가와 목표.** M3.5가 계산 buffer 낭비를 없애도 BERT F16 weight 약 64 MiB는 상주한다. 1 GB 모델이면 더 커진다. peak RSS가 전체 모델이 아니라 가장 큰 레이어 하나 크기에 가까워져야 한다. F16 결과는 M3와 cosine 0.9999 이상 같고, bge-small Q8_0 단일 추론은 peak RSS 40 MiB 미만이 목표다.

**가능한 방법.** 전체 RAM copy, 파일 전체 mmap 후 직접 참조, 레이어마다 `read`해 transient buffer에 copy, 레이어별 mmap view와 page 회수 hint가 있다.

**계획된 선택과 이유.** GGUF 파일은 mmap하되 persistent와 transient 영역을 나눈다. token embedding 등은 계속 두고 encoder weight는 현재 레이어만 사용하며 계산 후 `madvise(MADV_DONTNEED)`로 회수 hint를 준다. mmap은 파일 크기만큼 RAM을 즉시 쓰지 않고 필요한 page만 fault하며 명시적 read/copy를 줄인다. tensor가 mmap을 직접 안전하게 볼지 backend buffer로 copy할지는 M4 prototype에서 확정해야 한다.

**계획된 구현.** `LayerLoader`가 layer index를 weight 묶음으로 바꾸고, `StreamingRunner`가 embedding → 레이어 순회 → activation 교체 → pooling을 조정한다. persistent context, transient context와 이전/다음 activation buffer를 분리한다. 어느 순간에도 encoder layer 두 개를 동시에 상주시켜서는 안 된다.

**상태.** 미구현이다. `layer_loader.*`, `streaming_runner.*`가 없고 `use_streaming`은 오류로 거부된다.

### 8.2 Q8_0과 Q4_K_M

**무엇인가.** F16 weight는 값당 2 byte지만 int8/int4 계열은 block별 scale과 적은 bit로 근사 저장해 파일과 현재 레이어 RSS를 줄인다.

**왜 M4인가.** streaming만으로 전체 상주는 줄지만 embedding과 현재 레이어 크기는 dtype에 달렸다. 수십 MiB 목표에는 weight quantization이 함께 필요하다.

**가능한 방법.** load 때 F32로 dequantize, matmul 직전 임시 F32 생성, ggml quantized tensor와 native kernel 직접 사용이 있다.

**계획된 선택과 이유.** 세 번째 방식이다. 앞의 두 방식은 계산 순간 메모리를 다시 크게 만든다.

**검증 계획.** 인메모리 Harrier Q8_0은 sentence-transformers F32 oracle 대비 cosine
게이트가 이미 있다. M4에서는 llama.cpp embedding도 두 번째 oracle로 삼아 스트리밍
Q8_0/Q4의 max absolute difference와 cosine을 보고, F32 대비 품질 저하를 따로 기록한다.

**상태.** 인메모리 Q8_0 제품 graph는 자동 검증된다. Q4와 아직 없는 스트리밍
경로는 미검증이며, ggml이 타입을 지원한다는 사실만으로 맞다고 간주하지 않는다.

### 8.3 안정성과 bench

M4는 page fault/I/O를 의도적으로 바꾸므로 latency 증가만으로 실패라 하지 않는다. lifetime/window RSS, major/minor fault, disk read, cold/warm latency와 10,000회 후 file descriptor/VM region leak을 함께 본다. M4 baseline이 있어야 M5가 I/O 비용을 얼마나 amortize했는지 비교할 수 있다.

---

## 9. M5 — 레이어 단위 실제 배치

### 9.1 무엇인가

M4까지의 배치 API는 다음처럼 문장마다 전체 모델을 돌았다.

```text
문장 1: layer 0 → 1 → ... → 11
문장 2: layer 0 → 1 → ... → 11
```

M5는 순서를 바꾼다.

```text
layer 0: 문장 1, 2, ... N
layer 1: 문장 1, 2, ... N
...
```

### 9.2 왜 필요한가와 목표

M4는 문장마다 레이어 page를 다시 읽을 수 있다. N개가 한 번 load한 layer를 공유하면 I/O를 N개에 나눈다. batch 32에서 item당 latency를 M4 단일 호출의 1/10 이하로 낮추고 throughput을 높이는 것이 목표다.

### 9.3 가능한 방법과 선택

단일 호출을 여러 thread에서 병렬 실행, 모두를 최대 길이로 padding, 비슷한 길이끼리
bucket한 layer-wise batch, token packing이 있다. 첫 구현은 length bucketing +
padding이었지만 warm·혼합 길이에서 패딩 비용이 더 컸다. 최종 Harrier 경로는 실제
토큰을 이어 붙이고 문장별 offset으로 어텐션과 풀링 경계를 복원한다.

### 9.4 구현

- 공통 `BatchPlan`/`MaterializedBatch`: padded 배열과 packed 실제 토큰 배열, 길이·offset
- Harrier의 문장별 실제 길이 어텐션과 문장 구간별 mean/LAST pooling
- 임베딩, 선형변환, Norm과 FFN은 packed 실제 토큰만 처리
- BERT는 기존 right padding, attention mask와 mask-aware pooling 유지
- `max_batch` 초과 시 sub-batch. 별도 memory budget이나 숨은 retry 없음
- 원래 입력 순서로 output 복원
- 기본은 Harrier의 문장별 어텐션과 packed token-wise 연산. 공개 layout setter로
  기존 padded attention-mask 경로를 명시적으로 선택 가능
- `max_batch`는 자동 튜닝하지 않고 호출자가 workload와 메모리 예산에 맞춰 선택

eager는 sub-batch마다 한 graph를 만들고 gallocr를 관찰한 최대 shape까지 키운다.
streaming은 embedding row lease, 각 layer/group, final pooling을 sub-batch당 한 번씩
실행한다. diagnostics가 compute/lease 횟수, batch/item 및 valid/padding token 수를
기록한다.

**상태.** Harrier는 완료됐다. 문장별 어텐션 이후 F32/Q8 모두 sequential 결과와
최대 절대 오차 0으로 일치한다. BERT F16은 패딩 경로를 유지해 최대 `2.35e-4` 차이가
남아 있고 패킹 성능 이득도 받지 못한다.

초기 Docker Desktop 측정의 성능 gate 실패는 패딩 계산이 원인이었다. 패딩을 제거한
홈서버 x86_64 측정에서 Harrier Q8 warm·혼합 길이 처리량은 배치 1의 45.52에서 배치
10의 69.15문장/초로 51.9% 증가했고, 배치 64에서는 기존 패딩 방식보다 156% 높았다.
최종 `A문장별/F패킹/N패킹`의 파티션 비교는 `layer` 유지로 끝났다. `attn-ffn`은
차이를 구분할 수 없었고 `unit`은 모든 배치에서 4.5~8.4% 느리며 큰 배치의 전체
PSS도 더 컸다. 용어부터 재현 명령까지 [M5 최종 보고서](m5-overview.ko.md)에 있다.

---

## 10. M6 — 활성값 압축

### 10.1 KV cache가 아니라 activation

README의 과거 체크리스트에는 KV cache compression이라고 적혔지만 BERT형 encoder에는 autoregressive decoder의 KV cache가 없다. 최신 계획의 대상은 레이어 사이 hidden activation이다.

### 10.2 왜 필요한가와 목표

M4까지는 weight가 크지만 M5에서 batch가 커지면 F32 activation과 그래프 경계 버퍼가
커진다. 특히 `unit` 파티션은 가중치 임대량을 줄이고도 배치 64 PSS가 `layer`의
62.54 MiB보다 큰 89.95 MiB였다. 동일 RSS에서 더 큰 배치를 수용하는 것이 목표지만,
압축 전에 어떤 버퍼가 얼마나 차지하는지 같은 실행에서 분해해야 한다.

### 10.3 가능한 방법과 선택

F32 유지, F16/BF16, tensor 전체 scale int8/int4, row/token별 scale, TurboQuant 같은 변환 기반 기법이 있다. 첫 후보는 per-row absmax int8이다. 구현이 단순하고 다른 row의 outlier 영향도 줄인다. int4나 복잡한 기법은 int8 측정 후 판단한다.

### 10.4 계획된 구현과 검증

레이어 출력 직후 quantize해 값과 scale을 `ActivationStore`에 두고 다음 레이어 직전 dequantize한다. config는 `none/int8/int4`를 계획한다. cosine만이 아니라 STS-B Spearman 저하 0.01 이내, RSS와 round-trip overhead 15% 미만을 함께 본다.

**상태.** 미구현이고 알고리즘도 최종 확정되지 않았다. M5에서 큰 배치의 익명
메모리와 `unit` 경계 비용을 확인해 진입 근거는 생겼다. 다음 단계는 먼저
`slot_resident_bytes`, activation copy, graph replan과 가중치 lease high-water를
벤치 결과에 노출한 뒤, 공유 arena·backend 직접 전달·활성값 압축을 각각 비교한다.

---

## 11. M7 — 패키징과 C++ 래퍼

### 11.1 설치와 `find_package`

**무엇이며 왜 필요한가.** 지금은 이 저장소 안에서 직접 CMake를 실행해야 한다. 다른 프로젝트가 설치된 NanoEmbed를 `find_package(NanoEmbed)`와 `target_link_libraries`로 쓰게 해야 한다.

**가능한 방법과 선택.** 저장소를 subdirectory로 포함, pkg-config만 제공, CMake config와 pkg-config 모두 제공이 있다. CMake 사용자와 다른 build system을 모두 위해 두 형식을 계획한다.

### 11.2 shared library와 SOVERSION

정적 library는 최종 실행 파일에 포함되지만 shared library는 동적으로 load할 수 있어 언어 binding에 유리하다. `BUILD_SHARED_LIBS=ON`과 SOVERSION으로 ABI 호환 범위를 파일 이름에도 표현할 계획이다.

### 11.3 C++ RAII wrapper

**무엇이며 왜 필요한가.** C 사용자는 free를 직접 호출해야 한다. C++에서는 얇은 `.hpp`가 model/context pointer를 destructor에서 해제하면 예외와 early return에서도 안전하다.

**가능한 방법과 선택.** 별도 C++ API를 크게 설계할 수도 있지만 C API와 동작이 달라질 수 있다. C ABI를 호출하는 약 30줄의 얇은 wrapper를 계획해 C ABI를 source of truth로 유지한다.

**검증과 상태.** 별도 consumer project가 설치 package만으로 순수 C 예제를 빌드하고 CLI와 같은 결과를 내야 한다. install rule, package config, `.hpp`, shared SOVERSION은 모두 미구현이다.

---

## 12. M8 — Node.js binding

### 12.1 무엇이며 왜 필요한가

`npm install nanoembed` 후 문자열을 `Float32Array`로 바꾸고, 긴 추론이 Node event loop를 막지 않게 하는 native addon이다.

### 12.2 가능한 방법과 선택

CLI child process, Node FFI, WebAssembly, Node-API addon이 있다. CLI는 process/I/O 비용이 크고 FFI는 사용자 환경의 ABI와 lifetime 관리에 의존한다. WASM은 배포성이 좋지만 mmap과 native backend라는 핵심 경로를 그대로 쓰기 어렵다. V8 버전에 직접 묶이지 않고 C++ lifetime을 감싸기 쉬운 Node-API와 `node-addon-api`를 계획했다.

### 12.3 계획된 구현

- JS `Embedder` class
- `embed(string): Float32Array`
- `embedBatch(string[]): Promise<Float32Array[]>`
- worker thread로 event loop 비차단
- macOS arm64, Linux x86_64/arm64 prebuild
- prebuild가 없으면 source build fallback

한 thread-unsafe context를 여러 async 요청이 동시에 쓰지 않도록 context pool이나 직렬화가 필요하다. model weight는 공유하고 scratch는 context별로 둔 현재 C ABI가 기반이다.

**상태.** 미구현이며 `bindings/node/`와 npm package가 없다.

---

## 13. 마일스톤 완료의 공통 기준

파일이 생겼다는 사실만으로 완료하지 않는다. 다음을 모두 통과해야 한다.

1. 단위·통합 테스트
2. 해당 단계의 정확도 oracle
3. 이전 baseline 대비 의도하지 않은 메모리·성능 회귀 없음
4. 공개 C API를 쓰는 CLI smoke test

| 변경 | 대표 위험 | 검증 |
|---|---|---|
| tokenizer | token ID 불일치 | HF token fixture |
| forward | 레이어 수치 drift | activation fixture |
| 전체 조립 | 최종 embedding 오류 | golden cosine |
| allocator | OOM, 재사용 오염 | 길이/반복 테스트와 RSS |
| streaming | page leak, I/O 폭증 | RSS/fault/read/10k 반복 |
| batching | padding/mask 오류 | 단일 대 배치 비교 |
| compression | 검색 품질 저하 | STS-B 품질 지표 |
| packaging | ABI/설치 오류 | 외부 consumer project |
| Node | event loop 정지 | 비동기 응답성 테스트 |

## 14. 현재 가능한 것과 불가능한 것

현재 가능한 것:

- macOS/Linux build와 test
- 두 계열의 GGUF 검사와 load — `bert`(bge-small F16), `gemma3`(harrier-270m F32/양자화)
- WordPiece와 SentencePiece 계열 BPE tokenization, HF와 ID 완전 일치
- mean/CLS/last 단일 embedding. 기본값은 모델이 학습된 pooling
- 다국어 입력 (한국어·일본어·중국어·키릴·아랍어·이모지)
- C batch API. 단, 내부는 N회 단일 실행
- 최대 길이 제한과 truncate
- 두 모델 모두에 대한 layer별 activation 대조와 sentence-transformers 정확도 검사
- Linux M3/M3.5 memory/performance 비교

아직 불가능한 것:

- layer streaming과 제품 수준 Q8/Q4 (수치만 기록했고 streaming 경로 미검증)
- 실제 layer-wise batch와 activation compression
- padding이 있는 batch에서의 last-token pooling. 현재 pooling은 B=1 전제다
- EuroBERT/Jina v5 Nano와 GPT-2 BPE (식별만)
- 설치 가능한 shared package와 공개 C++ wrapper
- Node.js 호출

## 15. 권장 다음 순서

1. Linux 머신에서 `bench/baseline/M3.6.json`을 만든다. BERT 단문·장문과 Harrier
   F32 세 시나리오를 같은 머신에서 측정한다.
2. M4의 memory 전략을 prototype으로 검증한다. 여기서 확인해야 할 핵심은
   `mmap` + `madvise`로 token embedding table의 상주 page를 실제로 제어할 수
   있는지다. harrier는 parameter의 63%가 이 table이고 layer가 아니라서
   "읽고 버리기"로는 줄일 수 없다.
3. LayerLoader와 StreamingRunner를 구현한다.
4. 양자화 파일을 streaming 경로에서 다시 검증한다.

핵심은 새 모델 수학과 streaming을 동시에 디버깅하지 않는 것이다. M3.6이 인메모리
oracle을 확정했으므로, M4는 결과를 그대로 둔 채 weight lifetime만 바꾼다. 결과가
달라지면 원인은 streaming 쪽이라고 단정할 수 있다.

## 16. 용어 정리

| 용어 | 뜻 |
|---|---|
| embedding | 텍스트를 의미를 담은 고정 길이 float vector로 바꾼 결과 |
| GGUF | 모델 metadata와 tensor를 저장하는 파일 형식 |
| tensor | 다차원 숫자 배열. weight와 activation 모두 tensor다 |
| weight | 학습된 모델 parameter. 추론 중 읽기 전용 |
| activation | 입력이 레이어를 지날 때 생기는 중간 계산값 |
| context | 설정과 계산 buffer를 가진 실행 instance |
| ABI | 컴파일된 프로그램 사이의 호출과 data 배치 규약 |
| mmap | 파일을 virtual memory에 연결해 필요한 page를 OS가 읽게 하는 방식 |
| RSS | process가 현재 실제 RAM에 올려 둔 page 총량 |
| PSS | 공유 page 비용을 process 사이에 나눈 memory |
| USS | 해당 process만 독점하는 memory |
| page fault | 필요한 virtual page를 OS가 RAM에 준비하는 사건 |
| pooling | token hidden state를 문장 vector 하나로 줄이는 연산 |
| L2 normalization | vector 길이가 1이 되도록 나누는 연산 |
| quantization | float를 적은 bit 정수와 scale로 근사 저장하는 기법 |
| oracle/golden | 구현 결과와 비교할 기준 정답 |
| baseline | 특정 단계의 성능·memory 비교 기준 |
| façade | 여러 내부 구성요소를 단순한 interface 뒤에 감추는 객체 |
| RAII | C++ 객체 수명에 자원 획득과 해제를 묶는 패턴 |
