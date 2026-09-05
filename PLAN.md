# NanoEmbed 구현 계획

이 문서는 NanoEmbed가 해결하려는 문제, 현재 구현 상태, 앞으로 구현할 기능을 설명한다. 완료된 작업도 남겨 두는 이유는 다음 단계가 왜 필요한지 이해하려면 이전 구조와 측정 결과를 알아야 하기 때문이다.

상태 표기는 다음과 같다.

- **완료**: 현재 코드와 테스트에 반영되어 있다.
- **일부 완료**: 구조나 기반 코드는 있지만 사용자 기능 전체가 동작하지 않는다.
- **예정**: 아직 구현하지 않았다.
- **가정**: 구현 전에 실제 모델이나 측정으로 확인해야 한다.

## 1. 프로젝트 목표

NanoEmbed는 텍스트 임베딩 모델을 C++에서 실행하는 라이브러리다. 최종 목표는 100 MB에서 수 GB에 이르는 모델을 수십 MB 수준의 실제 메모리로 실행하는 것이다.

일반적인 인메모리 실행기는 모델 가중치를 모두 RAM에 올린다. 모델 파일이 커지면 가중치만으로 메모리 예산을 넘는다. NanoEmbed는 GGUF 파일에서 필요한 레이어의 가중치만 차례로 읽어 이 문제를 해결하려 한다.

첫 번째 대상 모델은 `bge-small-en-v1.5`다.

| 항목 | 값 |
|---|---:|
| 모델 구조 | BERT 인코더 |
| 인코더 레이어 | 12개 |
| 임베딩 차원 | 384 |
| 어텐션 헤드 | 12개 |
| FFN 중간 차원 | 1,536 |
| 어휘 크기 | 30,522 |
| 최대 입력 길이 | 512토큰 |
| F16 가중치 | 약 64 MiB |

이 모델은 충분히 작아 기준 구현을 만들기 쉽지만, 모든 가중치를 계속 메모리에 두면 목표인 수십 MB 예산을 안정적으로 맞추기 어렵다.

## 2. 메모리를 이해하기 위한 기본 용어

메모리 최적화 계획을 이해하려면 모델 실행 중 어떤 데이터가 메모리를 사용하는지 먼저 구분해야 한다.

### 2.1 모델 가중치

가중치는 학습이 끝난 모델의 숫자다. 토큰 임베딩 표, 어텐션의 Q·K·V 행렬, FFN 행렬, 정규화 계수 등이 여기에 포함된다.

추론 중에는 값이 바뀌지 않는다. 기본 eager 모드는 GGUF에서 읽은 모든 가중치를
모델 핸들이 살아 있는 동안 메모리에 유지한다. Linux의 M4 streaming 모드는 같은
가중치를 read-only mmap에서 직접 읽고 레이어별 페이지 상주를 조절한다.

### 2.2 계산 그래프

ggml은 연산을 호출하는 즉시 계산하지 않는다. 먼저 어떤 텐서에 어떤 연산을 적용할지 기록해서 계산 그래프를 만든다.

NanoEmbed의 그래프에는 다음과 같은 연산이 순서대로 들어간다.

```text
토큰 ID
  → 토큰·위치·타입 임베딩
  → BERT 인코더 블록 12개
  → 풀링
  → L2 정규화
  → 최종 임베딩
```

계산 그래프에는 두 종류의 메모리가 필요하다.

- **텐서 메타데이터**: 텐서의 자료형, 차원, stride와 연산 관계를 기록하는 작은 구조체다.
- **텐서 데이터**: 입력과 중간 계산 결과를 실제로 저장하는 바이트 영역이다. 중간 결과를 활성값이라고도 한다.

그래프 메모리는 모델 가중치와 다르다. 가중치는 학습된 고정 데이터이고, 그래프 메모리는 한 번의 추론을 실행하기 위해 만든 입력과 중간 결과다.

### 2.3 RSS

RSS(Resident Set Size)는 프로세스가 현재 실제 RAM에 올려 둔 메모리 크기다. 파일을 `mmap`해서 큰 가상 주소 영역을 확보해도 실제로 접근한 페이지만 RSS에 포함된다.

NanoEmbed가 엣지 장치에서 사용할 수 있는지를 판단할 때는 파일 크기나 가상 메모리보다 RSS가 중요하다. 장치에는 RSS만큼의 실제 여유 RAM이 필요하기 때문이다.

### 2.4 GGUF와 ggml

GGUF는 모델 파일 형식이다. 한 파일 안에 모델 구조를 설명하는 메타데이터, 토크나이저 정보, 텐서의 이름과 자료형, 실제 가중치가 들어 있다.

ggml은 GGUF를 읽고 텐서 계산 그래프를 실행하는 라이브러리다. NanoEmbed는 ggml의 파일 파서, 텐서 자료형, 그래프, CPU 연산 커널과 그래프 할당기를 사용한다.

NanoEmbed가 직접 책임지는 부분은 다음과 같다.

- GGUF에 적힌 모델 구조와 토크나이저 종류 선택
- BERT에 필요한 메타데이터와 텐서 검증
- WordPiece 토큰화
- 임베딩, 어텐션, FFN, 풀링 그래프 구성
- 공개 C API와 객체 수명 관리
- 테스트 기준 데이터와 성능 측정 도구
- 앞으로 추가할 레이어별 가중치 로딩과 배치 실행

## 3. 메모리를 줄이는 세 단계

### 3.1 그래프 버퍼 재사용 — 완료, M3.5

M3의 첫 구현은 `embed()`를 호출할 때마다 256 MiB 연속 메모리 영역을 만들었다. 당시 목표는 최적화보다 정확한 BERT 결과를 먼저 만드는 것이었다.

구현은 단순했다. `ggml_context`가 그래프에 등장하는 텐서의 데이터 공간을 앞에서부터 차례로 배치했다. 한 중간 텐서의 계산이 끝나도 그래프가 끝날 때까지 그 공간을 다른 텐서가 사용하지 않았다.

256 MiB는 BERT 계산이 수학적으로 반드시 요구하는 크기가 아니었다. 그래프에 필요한 최대 크기를 정밀하게 계산하지 않고, 일반 입력이 실행될 수 있도록 보수적으로 선택한 고정 용량이었다.

이 방식에는 세 가지 문제가 있었다.

1. 실제 계산에 쓰지 않는 공간까지 초기화해서 RSS에 포함됐다.
2. 호출할 때마다 256 MiB를 만들고 초기화해 지연 시간이 늘었다.
3. 길이 512 입력에는 256 MiB도 부족해 프로세스가 `GGML_ASSERT`로 종료될 수 있었다.

M3.5에서는 `ggml_gallocr`를 도입했다. 이 할당기는 그래프를 분석해 중간 텐서가 마지막으로 사용되는 시점을 찾는다. 사용이 끝난 텐서의 공간을 뒤에 실행할 텐서가 다시 사용할 수 있다.

현재 구조에서는 각 `nanoembed_context`가 `ComputeScratch`를 가진다. 이 객체가 CPU 백엔드, 그래프 할당기, 텐서 메타데이터 영역과 활성값 버퍼를 소유한다. 같은 컨텍스트에서 여러 번 추론하면 버퍼를 다시 사용한다.

### 3.2 레이어별 가중치 스트리밍 — 완료, M4

BERT 인코더의 각 레이어는 서로 다른 가중치를 사용하지만 실행 순서는 0번부터 마지막 레이어까지 고정되어 있다. 한 레이어의 계산이 끝나면 이전 레이어 가중치는 같은 요청에서 다시 필요하지 않다.

가중치 스트리밍은 이 성질을 이용한다.

```text
0번 레이어 가중치 읽기 → 0번 레이어 계산 → 가중치 해제
1번 레이어 가중치 읽기 → 1번 레이어 계산 → 가중치 해제
...
마지막 레이어 가중치 읽기 → 계산 → 가중치 해제
```

모델 파일 전체는 `mmap`으로 가상 주소 공간에 연결하되, 한 시점에는 현재 레이어의 페이지만 RSS에 남기는 것이 목표다. 토큰 임베딩처럼 모든 요청에서 필요한 공통 가중치와 현재 레이어 가중치만 실제 메모리에 남긴다.

여기서 M3.6이 확인한 사실 하나가 설계를 좌우한다. `harrier-270m`은 파라미터의
**63%가 토큰 임베딩 표**(262,144 × 640)다. 이건 레이어가 아니라서 위 순서대로
갈아끼울 수 없다. bge-small은 35%라 잘 드러나지 않던 성질이다.

```text
harrier-270m (F32 1.09 GB)
  token_embd     약 671 MB   63%   레이어 아님, 스트리밍 대상 아님
  18개 블록      약 402 MB   37%   블록당 약 22 MB
```

즉 "레이어를 버퍼에 복사했다 해제한다" 방식으로는 671 MB가 그대로 남아 M4의
목표를 달성할 수 없다. 대신 임베딩 표는 `ggml_get_rows`가 입력에 등장한 토큰의
행만 읽는다는 성질을 쓴다. 30토큰짜리 입력이면 262,144행 중 30행만 건드리므로,
`mmap`으로 걸어두면 실제로 만진 페이지만 상주한다.

이것이 M4를 "명시적 로드/해제"가 아니라 "`mmap` + `madvise`로 페이지 상주 제어"로
설계해야 하는 이유다. 대상 모델을 열어보기 전에는 알 수 없는 사실이었다.

현재 구현은 하나의 `MAP_PRIVATE|PROT_READ` mapping에 metadata-only ggml weight
tensor를 연결한다. 토큰 임베딩은 실제 token ID가 가리키는 row의 page range만,
각 transformer block은 해당 레이어 range만 `MADV_WILLNEED` lease로 보호한다.
계산과 다음 활성값 복사가 끝나면 마지막 사용자가 `MADV_DONTNEED`를 요청한다. 서로
다른 streaming context가 공유 mapping을 동시에 사용할 때는 active lease와 common
range를 coordinator가 보호하며, advice 실패나 mixed mode는 eager fallback 없이 오류다.

레이어 사이의 활성값은 context-local F32 ping-pong buffer 두 개로 유지한다. 따라서
M4는 가중치 상주를 줄이지만 activation memory까지 제거하지는 않는다. `madvise`는
커널에 대한 advisory이고 실제 page eviction을 보장하지 않으므로, 기능의 정확성과
저장된 RSS/PSS/USS 관찰을 구분해서 해석한다.

양자화로 우회할 수도 없다. 배포된 q8_0 / q5_k / q4_k 세 파일 모두 토큰 임베딩 표를
q8_0으로 유지하고 블록만 더 줄인다. 그래서 임베딩 표는 어느 파일에서든 178 MB로
고정이고, q8_0에서 q4_k로 내려가도 파일 전체는 17%밖에 줄지 않는다. 임베딩 표를
줄이는 방법은 정밀도를 낮추는 것이 아니라 **필요한 행의 페이지만 상주시키는 것**이다.

### 3.3 레이어 단위 배치 — 완료

스트리밍은 레이어를 바꿀 때마다 파일 페이지를 읽는다. 문장마다 모든 레이어를 다시 읽으면 I/O 비용이 반복된다.

M5에서는 여러 문장을 같은 레이어에서 함께 계산한다.

```text
0번 레이어 한 번 로드 → 문장 32개의 0번 레이어 계산
1번 레이어 한 번 로드 → 문장 32개의 1번 레이어 계산
...
```

현재 eager와 Linux streaming 모두 실제 배치 그래프를 사용한다. 모든 입력을 토큰화해
길이순 stable sort하고 `max_batch` 단위로 나눈 뒤, Harrier는 실제 토큰만 이어 붙인
배열과 문장별 offset을 사용한다. 어텐션과 풀링은 문장 경계를 따르고 결과는 원래 입력
순서로 복원한다. BERT는 아직 right padding과 attention/pooling mask 경로를 유지한다.

한 번 읽은 가중치를 여러 문장이 공유하므로 I/O lease 횟수는 item 수가 아니라
sub-batch 수에 비례한다. 이 성질은 측정으로 확인됐다 — cold 캐시에서 문장당
major fault가 정확히 1/B로 준다 (263.7 → 25.9, 배치 10).

처음 측정에서는 손익이 캐시 상태와 입력 길이에 따라 갈렸다. 디스크 읽기가 0회인
warm 조건에서 길이가 섞이면 아낄 I/O는 없는데 패딩 비용만 남아 27% 느려졌다.

**패딩을 제거해 이 문제를 해소했다.** 두 단계였다. 어텐션을 문장별로 분리하고,
이어서 모든 토큰 단위 연산에서 패딩을 뺐다. 후자가 가능한 이유는 선형 변환·FFN·
정규화가 토큰 하나를 받아 토큰 하나를 내놓기 때문이고, 문장 간 간섭이 가능한
어텐션은 앞 단계에서 이미 분리했기 때문이다.

| 조건 | 패딩 제거 전 | 패딩 제거 후 |
|---|---:|---:|
| warm + 길이 편차 큼 | −27% | **+51.9%** |
| warm + 길이 균일 | +56% | +56% (경로 미변경) |
| cold | +523% | **+607%** |

정확도도 해결됐다. 마스크가 붙으면 softmax가 패딩 칸까지 훑어 덧셈 순서가 달라졌는데,
실제 길이만 계산하니 harrier는 순차 처리와 **완전히 동일**해졌다. 원래 기준(1e-5)을
통과한다.

남은 것은 BERT다. 아직 마스크 경로를 쓰므로 최대 오차 2.345e-4가 그대로이고 성능
이득도 받지 못한다. 측정, 구현과 최종 결정은
[`docs/m5-overview.ko.md`](docs/m5-overview.ko.md)에 있다.

### 3.4 활성값 압축 — 예정, M6

배치가 충분히 커지면 가중치보다 문장별 활성값이 더 많은 메모리를 사용할 수 있다. M6에서는 레이어 사이에 보관하는 활성값을 int8 또는 int4로 압축하는 방안을 검토한다.

이 기능은 스트리밍과 실제 배치가 동작한 뒤 측정해야 필요성과 효과를 판단할 수 있으므로 후순위다.

## 4. M3.5를 M4와 분리한 이유

그래프 메모리 개선과 가중치 스트리밍은 서로 다른 메모리 영역을 줄인다. 두 변경을 한 번에 적용하면 어느 변경이 얼마나 기여했는지 알기 어렵다.

M3의 짧은 문장 시나리오에서 측정한 최대 RSS는 330.05 MiB였다. 대략적인 구성은 다음과 같다.

```text
M3 최대 RSS 약 330 MiB
= 그래프용 고정 메모리 256 MiB
+ 모델 가중치 약 64 MiB
+ 토크나이저, 메타데이터, 프로그램 메모리 약 10 MiB
```

가중치 스트리밍의 효과가 왜 작게 보이는지 상한을 계산해 볼 수 있다.

다음은 설명을 위한 가정이다. 가중치 스트리밍이 64 MiB 모델 가중치를 전부 없앤다고 가정한다. 실제 실행에는 토큰 임베딩과 현재 레이어 가중치가 필요하므로 이 정도로 줄일 수는 없다.

```text
변경 전: 그래프 256 + 가중치 64 + 기타 10 = 330 MiB
변경 후: 그래프 256 + 가중치  0 + 기타 10 = 266 MiB
```

감소율을 계산하면 다음과 같다.

```text
(330 - 266) / 330 × 100 ≈ 19.4%
```

가중치를 전부 없앤다는 상한 가정에서도 전체 RSS는 약 19%만 줄어든다. 256 MiB 그래프 메모리가 그대로 남기 때문이다. 실제 가중치 스트리밍은 공통 임베딩과 현재 레이어 가중치를 유지하므로 감소율이 이보다 작다.

따라서 가중치 스트리밍을 평가하기 전에 그래프 메모리 할당 방식을 먼저 고쳐야 했다. 이 작업을 M3.5로 분리했다.

M3.5에서 길이 512를 위해 예약한 활성값 버퍼는 약 15.76 MiB다. 짧은 문장 시나리오의 최대 RSS는 330.05 MiB에서 76.35 MiB로 줄었다.

| 지표 | M3 | M3.5 | 변화 |
|---|---:|---:|---:|
| 최대 RSS | 330.05 MiB | 76.35 MiB | -76.9% |
| p50 지연 시간 | 240.05 ms | 41.97 ms | -82.5% |
| 초당 처리 문장 | 4.00 | 20.27 | +407% |
| 측정 구간 minor page fault | 327,685,001 | 0 | 반복 초기화 제거 |

이 값은 `bench/baseline/M3.json`과 `bench/baseline/M3.5.json`에 기록된 같은 Linux 머신의 실측값이다. 지연 시간 개선은 BERT 계산량이 줄어서가 아니다. 호출마다 256 MiB를 초기화하던 비용이 사라졌기 때문이다.

길이 512 입력의 M3.5 최대 RSS는 93.01 MiB였다. 짧은 입력보다 활성값이 크기 때문에 RSS도 더 크다. 그래프 할당기가 예약한 가상 공간 전체가 항상 RSS가 되는 것은 아니며, 실제로 접근한 페이지가 주로 상주한다.

## 5. 현재 소프트웨어 구조

현재 의존성은 공개 API에서 ggml 쪽으로 한 방향으로 흐른다.

```text
사용자 프로그램
    ↓
include/nanoembed/nanoembed.h        공개 C API
    ↓
src/api/c_api.cpp                    인자 검사와 오류 변환
    ↓
src/embedder.cpp                     모델과 컨텍스트 조정
    ├─ src/arch/*                    모델 구조 선택과 그래프 구성
    ├─ src/tokenizer/*               토크나이저 선택과 실행
    ├─ src/forward/*                 BERT 연산 그래프 구성
    └─ ggml                          텐서 할당과 CPU 계산
```

### 5.1 모델 구조 선택

`create_model_arch()`는 GGUF의 `general.architecture` 값을 읽는다.

- `bert`이면 `BertModelArch`를 만든다.
- `gemma3`이면 `Gemma3ModelArch`를 만든다.
- `eurobert`이면 종류를 알아본 뒤 미구현이라는 오류를 반환한다.
- 그 밖의 값이면 지원하지 않는 구조라고 보고한다.

`ModelArch`는 모델마다 달라지는 정보를 묶는 내부 인터페이스다. 메타데이터 키, 필요한 입력 텐서, 가중치 이름, 인코더 그래프 구성이 여기에 속한다.

### 5.2 토크나이저 선택

`create_tokenizer()`는 `tokenizer.ggml.model`을 별도로 읽는다. 모델 구조와 토크나이저는 반드시 같은 이름을 사용하지 않으므로 독립적으로 선택한다.

- `bert` 토크나이저는 WordPiece다.
- `llama` 토크나이저는 SentencePiece 계열 BPE다. 이 태그는 어휘 형식을 가리킬 뿐
  알고리즘을 정하지 않는다. 어느 쪽인지는 파일이 무엇을 싣고 있는지로 갈린다.
  의미 있는 `scores`가 있으면 unigram, `merges`가 있으면 순위 기반 BPE다.
  harrier는 둘 다 싣고 있지만 `scores`가 `0,1,2,…`라는 토큰 인덱스일 뿐이므로 BPE다.
- `gpt2` 토크나이저는 byte-level BPE임을 알아보지만 아직 구현하지 않았다.

### 5.3 모델과 컨텍스트의 소유권

공개 API에는 모델과 컨텍스트라는 두 불투명 핸들이 있다.

| 객체 | 주요 소유 데이터 | 공유 가능 여부 |
|---|---|---|
| `nanoembed_model` | GGUF 핸들, 가중치, 모델 구조, 토크나이저 | 읽기 전용으로 여러 컨텍스트가 공유 가능 |
| `nanoembed_context` | 실행 설정, CPU 백엔드, 그래프 할당기, 재사용 활성값 버퍼 | 한 컨텍스트를 동시에 호출하면 안 됨 |

컨텍스트별로 `ComputeScratch`를 두기 때문에 서로 다른 컨텍스트는 같은 모델을 사용하면서 동시에 실행할 수 있다. 변경 가능한 실행 버퍼를 모델에 두지 않는 것이 핵심이다.

## 6. 공개 C API

공개 인터페이스는 [`include/nanoembed/nanoembed.h`](include/nanoembed/nanoembed.h)에 있다. C ABI를 유지하기 위해 구현 클래스와 ggml 타입은 외부에 노출하지 않는다.

기본 호출 순서는 다음과 같다.

```text
nanoembed_load_model
    → nanoembed_new_context
        → nanoembed_embed 또는 nanoembed_embed_batch
    → nanoembed_free_context
→ nanoembed_free_model
```

`nanoembed_embed_batch()`는 실제 배치 그래프를 만든다. 길이순 정렬은 내부 계산
순서만 바꾸며 출력은 호출자가 넘긴 순서로 복원한다. `max_batch`보다 많은 입력은
여러 sub-batch로 나뉘고, 실패 시 더 작은 batch나 eager 경로로 재시도하지 않는다.
Harrier는 기본적으로 `A문장별/F패킹/N패킹`을 사용한다. dense padded kernel이 더
유리한 workload를 위해 `nanoembed_context_set_batch_layout()`으로
`NANOEMBED_BATCH_LAYOUT_PADDED`를 선택할 수 있다. 보편적인 최적 batch size를
가정하지 않으며 호출자가 `max_batch`를 workload와 메모리 예산에 맞춘다.

기본 컨텍스트 설정은 다음과 같다.

| 설정 | 기본값 | 현재 의미 |
|---|---:|---|
| `n_threads` | 0 | 자동 선택. macOS 하이브리드 CPU는 성능 코어 수를 우선 사용 |
| `max_batch` | 64 | 한 실제 sub-batch의 최대 문장 수. 초과 입력은 내부 분할 |
| `max_seq_len` | 512 | 2 이상. 더 긴 입력은 자르고 모델 자체 상한도 넘지 않음 |
| `use_streaming` | 0 | Linux에서 1은 strict mmap/layer streaming; 그 밖의 값과 비-Linux 1은 오류 |
| `pooling` | Model default | 필요하면 Mean/CLS/LAST로 명시적 변경 가능 |
| `normalize` | 1 | 결과에 L2 정규화 적용 |

`batch_layout`은 frozen context parameter 구조체에 필드를 추가하지 않고 별도 setter로
고른다. 기본값은 모델이 지원하는 최적화 경로이며 `PADDED`는 right padding, attention
mask와 padded token-wise 연산을 강제한다.

내부 C++ 예외는 C API 경계에서 잡아 상태 코드와 `nanoembed_last_error()` 문자열로 바꾼다. 예외가 C ABI 밖으로 나가지 않는다.

## 7. 정확도 검증

최적화 전후 결과가 같은지 확인하려면 비교 기준이 필요하다. NanoEmbed는 서로 다른 범위의 테스트 기준 데이터를 사용한다.

### 7.1 토큰 ID 기준 데이터

Hugging Face 토크나이저가 만든 토큰 ID를 TSV로 저장한다. NanoEmbed WordPiece 결과와 원소별로 비교한다. 토큰화가 다르면 뒤의 모든 계산도 달라지므로 가장 먼저 확인할 경계다.

### 7.2 레이어별 활성값 기준 데이터

Hugging Face에서 임베딩 레이어와 각 BERT 블록의 출력을 저장한다. NanoEmbed의 그래프 빌더 결과와 비교하면 처음 차이가 생긴 레이어를 찾을 수 있다.

### 7.3 최종 임베딩 기준값

`golden_test.cpp`라는 파일명은 유지하지만 문서에서는 기준값 테스트라고 부른다. 문장별 최종 임베딩을 sentence-transformers 결과와 비교한다.

| 기준 | 요구값 |
|---|---:|
| 각 문장의 코사인 유사도 | 0.9999 이상 |
| 전체 평균 코사인 유사도 | 0.99999 이상 |

sentence-transformers는 FP32 PyTorch 계산을 사용하고 NanoEmbed는 F16 가중치와 ggml 커널을 사용하므로 비트 단위로 같을 필요는 없다. 코사인 유사도는 두 구현이 의미상 같은 벡터를 만드는지 확인하는 기준이다.

## 8. 성능 측정 방법

성능 측정 도구는 Linux 전용이다. `/proc/<pid>/status`, `statm`, `smaps_rollup`, `clear_refs`에서 메모리 정보를 읽기 때문이다.

### 8.1 부모 프로세스와 작업 프로세스

부모 프로세스는 모델을 로드하지 않는다. 자기 실행 파일을 `fork`한 뒤 `exec`해서 새 작업 프로세스를 만들고, 바깥에서 작업 프로세스의 `/proc` 정보를 관찰한다.

`fork`만 사용하면 부모가 이미 사용한 메모리 페이지가 copy-on-write 상태로 자식 RSS에 보일 수 있다. `exec`은 주소 공간을 새 프로그램 상태로 교체한다. 따라서 작업 프로세스는 측정 도구 자체가 사용한 메모리를 물려받지 않는다.

### 8.2 측정하는 값

| 값 | 의미 |
|---|---|
| RSS | 현재 실제 RAM에 올라온 전체 페이지 |
| PSS | 공유 페이지를 프로세스 수에 따라 나눈 값 |
| USS | 이 프로세스만 사용하는 페이지 |
| lifetime peak RSS | 모델 로드와 준비를 포함한 전체 생애 최대 RSS |
| window peak RSS | 반복 측정 구간의 최대 RSS |
| page fault | 아직 상주하지 않은 페이지에 접근한 횟수 |
| p50/p90/p99 | 개별 호출 지연 시간의 백분위수 |
| throughput | 1초에 처리한 문장 수 |

RSS에는 커널이 기록하는 최대값 `VmHWM`이 있다. PSS와 USS에는 같은 최대값이 없으므로 일정 간격으로 읽은 값 중 최대를 사용한다. JSON 키에 `_sampled`가 붙는 이유다.

### 8.3 비교 가능한 환경

CPU와 커널이 다르면 코드가 같아도 결과가 달라진다. 모든 결과는 커널, CPU 모델, 논리 CPU 수와 페이지 크기를 기록한다.

`bench/compare.py`는 기준 측정값과 현재 결과의 환경 정보를 비교한다. `--strict`에서는 환경이 다르면 종료 코드 2로 중단한다.

지연 시간은 같은 머신에서도 다른 작업의 영향을 크게 받는다. 메모리와 페이지 폴트는 상대적으로 안정적이다. 고정된 자체 호스팅 러너가 생기기 전까지 성능 기준값은 같은 개발 머신에서 수동으로 측정한다.

## 9. 마일스톤

### M1 — 공개 C ABI와 빌드 뼈대: 완료

- CMake 프로젝트와 ggml 서브모듈 구성
- `nanoembed_model`, `nanoembed_context` 불투명 핸들 정의
- 상태 코드와 thread-local 오류 메시지 정의
- 순수 C 코드로 헤더와 링크를 확인하는 `abi_link` 테스트 추가

### M2 — GGUF BERT 검사기: 완료

- BERT 메타데이터와 필수 텐서 이름·자료형·차원 크기 검증
- 가중치 데이터를 읽지 않는 검사 경로 구현
- 누락된 텐서나 잘못된 차원을 추론 전에 구체적인 오류로 보고

### M3 — 인메모리 BERT 임베더: 완료

- WordPiece 토크나이저 구현
- 임베딩, 어텐션, FFN, 풀링 그래프 구현
- C API와 CLI에서 실제 임베딩 생성
- 토큰 ID, 레이어 활성값, 최종 임베딩 기준 데이터와 테스트 구축
- Linux 성능 측정 도구와 M3 기준 측정값 기록

M3는 정확한 결과를 만드는 기준 구현이다. 메모리 최적화보다 각 계산 단계가 맞는지를 먼저 검증했다.

### M3.5 — 그래프 메모리 개선: 완료

- 호출마다 생성하던 고정 256 MiB 영역 제거
- `ggml_gallocr`로 중간 텐서 버퍼 재사용
- 컨텍스트별 `ComputeScratch`에 실행 버퍼 배치
- 길이 512 입력과 장문·단문 교차 실행 회귀 테스트 추가
- M3.5 기준 측정값 기록

### M3.6 — 모델 교체 구조와 두 번째 모델: 완료 (벤치 baseline 제외)

두 번째 모델은 `microsoft/harrier-oss-v1-270m`이다. Gemma 3 디코더를 임베딩
모델로 파인튜닝한 것으로, GGUF에는 `gemma3`로 선언된다. 원래 후보였던
`jina-embeddings-v5-text-nano`(`eurobert`)에서 바꾼 이유는 두 가지다.
jina는 cc-by-nc-4.0(비상업)이라 fixture와 baseline이 저장소에 박히면 되돌리기
어렵고, Harrier가 요구하는 메커니즘(GQA, causal, QK-norm, head_dim 분리)이
현대 decoder-only 모델의 표준이라 다음 모델로 그대로 이어진다.

완료한 것:

- `ModelArch` / `Tokenizer` 인터페이스와 두 개의 레지스트리
- 계열 무관 GGUF 헬퍼(`gguf_util`)와 `gemma3` 전용 스캐너
- 메커니즘 이름의 그래프 빌더: `rms_norm`, `rope`, `gated_ffn`, `gqa_attention`
- SentencePiece 계열 BPE 토크나이저
- 공개 API의 LAST 풀링과 "모델이 학습된 풀링" 기본값
- 두 계열 모두를 도는 토크나이저·활성값·golden·길이 경계 테스트
- 동일 모델의 서로 다른 두 context를 병렬 실행하는 동시성 테스트
- 최대 길이 예약 뒤 첫 추론에서 graph buffer가 커지지 않는 불변식 테스트
- Harrier Q8_0을 F32 oracle과 비교하는 자동 회귀 게이트
- 다국어 코퍼스 (`tests/corpus/eval_multilingual.txt`)

검증 결과. 같은 정밀도(F32)로 비교했을 때 레이어별 상대오차는 1e-6~2e-5이고,
임베딩 단계는 정확히 일치한다. 최종 임베딩은 sentence-transformers 대비
132문장 전부에서 코사인 1.000000이다.

양자화 손실은 F32 기준값 대비 따로 측정한다. 구현 정합성과 양자화 오차는
서로 다른 검사 항목이므로 하나의 임계값으로 묶지 않는다.

M3.6 벤치 선택에는 BERT 단문·장문과 Harrier F32가 모두 포함된다. 남은 것은
`bench/baseline/M3.6.json`이다. 벤치 도구는 Linux 전용이고 기존
baseline은 다른 머신에서 측정됐으므로, 같은 머신에 접근할 수 있을 때 별도로
기록한다.

### M4 — 레이어 스트리밍과 양자화 가중치: 완료

- read-only GGUF mmap과 validated metadata-only weight tensor
- token row, common range, 현재 layer range를 구분한 residency lease
- 레이어 계산 뒤 `madvise(DONTNEED)`와 동시 context 보호
- BERT F16, Harrier F32/Q8_0/Q4_K 실행 및 strict `use_streaming=1`
- 최초 context의 atomic mode lock, mixed mode와 비-Linux의 loud failure
- benchmark requested/resolved mode 증거와 eager/streaming full-vector 비교

동일 M4 binary의 Docker Desktop arm64 warm/profile-off 측정에서 streaming의 lifetime
RSS는 BERT short 79.07→14.90 MiB(-81.2%), BERT long 91.94→34.02 MiB(-63.0%),
Harrier F32 1111.97~1113.90→102.24~102.94 MiB(약 -90.8%), Harrier Q8_0
360.83~363.04→87.04~87.05 MiB(약 -76.0%)였다. Q4_K는 313.26→84.62
MiB(-73.0%)이지만 pre-M4 baseline이 없는 report-only 축이다. BERT short의 40 MiB
목표는 충족했고 long context와 Harrier는 activation과 common/token range 때문에 그보다
높다.

메모리 절감에는 비용이 있다. 같은 authoritative warm/profile-off 결과에서 throughput은
BERT long +0.1%, BERT short -3.4%, Harrier F32 -4.5~-17.2%, Harrier Q8_0
-21.7~-38.0%, Q4_K -14.0%였다. strict-cold canonical startup은 Harrier에서
12.9~48.7% 줄었지만 첫 layer 실행의 page fault와 inference-only latency는 증가했다.
출력은 F16/F32/Q8/Q4 모두 eager와 원소 수준에서 일치(max absolute error, MAE, RMSE,
norm difference 0)했고 기존 PyTorch cosine gate도 유지한 채 통과했다.

이 수치는 [저장된 B5 closeout](bench/results/M4-docker-desktop-arm64/CLOSEOUT.md)의 동일
Docker Desktop 4.38.0 Ubuntu 24.04 arm64 VM과 bind mount 범위에만 해당한다. 10 ms
PSS/USS는 sampled lower bound이고 profile-on latency는 diagnostic이다. 1회 독립 실행,
null confidence interval이며 통계적 유의성을 주장하지 않는다.

### M5 — 실제 레이어 단위 배치: 완료 (Harrier 범위, BERT 패킹 제외)

- 길이 stable sort, right padding, 원래 순서 복원 구현
- eager와 Linux streaming의 실제 B축 graph 구현
- padding-aware BERT/Gemma attention과 Mean/CLS/LAST pooling 구현
- `max_batch` 초과 입력의 내부 분할과 batch 단위 streaming lease 구현
- schema v3 batch 32/128 및 partition별 RSS/PSS/USS 측정 완료
- 실용 회귀 정확도와 phase-count gate 통과; 원 계획의 더 엄격한 정확도 gate는 실패
- 최초 측정에서 warm·혼합 길이 처리량 gate 실패 (배치 1→10 −27%)
- 원인을 패딩으로 확정하고 제거: 어텐션 문장별 분리 후 토큰 단위 연산 패킹
- 패딩 제거 후 최종 반복 측정에서 warm·혼합 길이 **+51.9%**로 gate 통과
- harrier 정확도가 순차 처리와 완전히 일치해 원 기준(1e-5) 통과
- 최종 `A문장별/F패킹/N패킹`에서 `layer`/`attn-ffn`/`unit` 재측정 완료
- Harrier 기본값을 위 구성으로 확정하고 padded/masked 호환 layout은 공개 옵션으로 유지
- batch size는 자동 최적화하지 않고 공개 `max_batch`로 호출자가 선택
- `layer`를 유지: `attn-ffn`은 사실상 동률, `unit`은 4.5~8.4% 느리고 큰 배치에서 메모리 증가
- BERT는 아직 마스크 경로 유지 — 최대 오차 2.345e-4, 성능 이득 미적용

용어, 구현, 정확도, 초기 실패, 패딩 제거, 홈서버 반복 측정, 파티션과 메모리 해석,
재현 명령은 단일 정본인
[`docs/m5-overview.ko.md`](docs/m5-overview.ko.md)에 있다. 초기 Docker Desktop 측정
원본과 당시 판정은
[`bench/results/M5-docker-desktop-arm64/CLOSEOUT.md`](bench/results/M5-docker-desktop-arm64/CLOSEOUT.md)에
보존한다.

### M6 — 활성값 압축: 예정, 후순위

- 레이어 사이 활성값의 int8 또는 int4 압축
- 압축에 필요한 scale 메타데이터 정의
- 압축 전후 정확도와 메모리·지연 시간 비교

배치 활성값이 실제 병목으로 측정된 경우에만 진행한다.

### M7 — 설치와 C++ 래퍼: 예정

- RAII 기반 C++ 헤더
- shared library 선택지
- `install`과 `find_package` 지원
- 버전과 패키지 메타데이터 정리

### M8 — Node.js 바인딩: 예정

- N-API 바인딩
- 작업 스레드에서 동기 C++ API 실행
- JavaScript Promise 인터페이스
- 모델과 컨텍스트 수명 관리

## 10. 마일스톤 통과 조건

다음 단계로 넘어가기 전에 네 가지를 확인한다.

1. **빌드와 API**: 지원 플랫폼에서 빌드되고 공개 C ABI 테스트가 통과한다.
2. **정확도**: 해당 단계가 책임지는 토큰·레이어·최종 임베딩 기준값 테스트가 통과한다.
3. **메모리와 성능**: 같은 머신에서 이전 기준 측정값과 비교하고 의도하지 않은 악화를 설명하거나 수정한다.
4. **문서**: 현재 동작, 아직 지원하지 않는 기능, 재현 명령을 함께 갱신한다.

기본 검증 명령은 다음과 같다.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Linux에서 성능 결과를 만들고 비교할 때는 다음 명령을 사용한다.

```sh
python bench/runner.py \
  --milestone M3.5 \
  --out bench/results/local.json

python bench/compare.py \
  bench/baseline/M3.json \
  bench/results/local.json
```

## 11. 남은 설계 질문

- M5의 warm 손해 원인이던 패딩은 Harrier에서 제거됐다. 최종
  `A문장별/F패킹/N패킹/P레이어`는 warm·혼합 길이 배치 1→10에서 +51.9%였고,
  배치 10부터 처리량이 거의 포화했다. BERT에는 같은 경로가 아직 적용되지 않았다.
- `unit`은 동시 가중치 임대량을 줄이지만 큰 배치의 경계 활성값을 F32 `SlotStore`에
  유지해 전체 PSS가 늘었다. 다음 메모리 단계에서는 lease와 slot 진단을 같은 실행에서
  노출하고 생존 구간 기반 공유 버퍼 또는 backend 직접 전달을 먼저 검토한다.
- sub-batch별 `S`, `B`, 패딩 토큰 수가 벤치마크 결과로 노출되지 않는다. 이 값 없이는
  배치의 손익도, 메모리가 배치 크기에 대해 단조롭지 않은 이유도 정량 설명이 불가능하다.
- Docker Desktop 결과를 물리 Linux target의 성능/저장장치 동작으로 일반화할 수 없다.
  자체 호스팅 target runner와 여러 independent run 운영 규칙이 필요하다.
- Harrier token embedding table은 layer가 아니며 요청 token row가 계속 필요하다. 실제
  multi-worker PSS sharing은 단일 프로세스 sampled PSS로 추정하지 말고 별도 실험한다.
- verified Hugging Face revision으로 golden fixture를 재생성하기 전까지 B5 정확도
  provenance는 `legacy_unverified`다.
- ggml 서브모듈을 올릴 때 mapped leaf tensor와 backend buffer 소유 규칙, quantized
  `ggml_mul_mat`/`ggml_get_rows`, `madvise` lease 경계를 다시 검증해야 한다.

라이선스와 배포 형식은 M7 패키징 단계 전에 확정한다.
