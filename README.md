# NanoEmbed

NanoEmbed는 텍스트를 의미 벡터로 바꾸는 C++17 라이브러리다. 서버뿐 아니라 메모리가 제한된 엣지 장치에서도 임베딩 모델을 실행하는 것이 목표다.

임베딩은 문장 하나를 고정 길이 실수 배열로 표현한 값이다. 의미가 비슷한 문장은 가까운 벡터가 되므로 검색, 추천, 중복 탐지, RAG의 검색 단계에 사용할 수 있다.

## 현재 구현 범위

현재 코드는 첫 번째 기준 구현과 그래프 메모리 개선을 마친 상태다.

- `bge-small-en-v1.5`의 GGUF 파일을 읽을 수 있다.
- BERT 구조와 WordPiece 토크나이저를 지원한다.
- C API로 단일 문장과 여러 문장을 임베딩할 수 있다. 여러 문장 API는 아직 내부에서 한 문장씩 처리한다.
- 중간 계산값을 저장하는 버퍼는 호출마다 새로 만들지 않고 컨텍스트별로 재사용한다.
- 모델 가중치는 아직 모두 메모리에 올린다. 레이어별 가중치 스트리밍은 M4에서 구현할 예정이다.
- `eurobert`와 GPT-2 계열 BPE 토크나이저는 GGUF 종류를 식별하지만 아직 실행하지 못한다.

지원 현황은 다음과 같다.

| 모델 구조 | 토크나이저 | 상태 |
|---|---|---|
| `bert` | `bert` WordPiece | 지원 |
| `eurobert` | `gpt2` byte-level BPE | 구조만 식별, 구현 예정 |

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

모델 파일이 없으면 모델이 필요한 테스트는 건너뛴다. 로컬에서 전체 정확도 테스트를 실행하려면 다음 경로에 모델을 둔다.

```text
models/bge-small-en-v1.5-f16.gguf
```

CI에서는 모델을 내려받은 뒤 `NANOEMBED_REQUIRE_MODEL=ON`으로 설정한다. 모델이 없어서 테스트가 조용히 건너뛰는 일을 막기 위한 설정이다.

## CLI 사용

표준 입력으로 문장을 전달하면 쉼표로 구분된 임베딩 벡터를 출력한다.

```sh
echo "hello world" \
  | ./build/bin/nanoembed-cli models/bge-small-en-v1.5-f16.gguf
```

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

벤치마크 도구는 Linux의 `/proc`에서 프로세스 메모리와 페이지 폴트 정보를 읽는다. 따라서 `nanoembed-bench`는 Linux에서만 빌드된다. 라이브러리와 일반 테스트는 macOS와 Linux에서 모두 빌드할 수 있다.

```sh
# 모델 없이 측정 도구 자체를 검사한다.
./build/bin/nanoembed-bench --selftest

# M3.5 시나리오를 실행해 결과 파일을 만든다.
.venv/bin/python bench/runner.py \
  --milestone M3.5 \
  --out bench/results/local.json

# M3 기준 측정값과 비교한다.
.venv/bin/python bench/compare.py \
  bench/baseline/M3.json \
  bench/results/local.json
```

성능 수치는 같은 머신에서 측정한 결과끼리만 비교해야 한다. CPU와 커널이 다르면 코드 변경보다 실행 환경 차이가 더 크게 나타날 수 있다. `compare.py`는 각 결과에 기록된 환경 정보를 확인하고, `--strict` 모드에서는 환경이 다르면 비교를 중단한다.

측정 지표와 메모리 수치의 의미는 [테스트와 벤치마크 가이드](docs/guide/26080401/07-tests-bench-workflow.md)에서 설명한다.
