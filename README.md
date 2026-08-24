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
- 모델 가중치는 아직 모두 메모리에 올린다. 레이어별 가중치 스트리밍은 M4에서 구현할 예정이다.

정확도는 sentence-transformers와의 코사인 유사도로 검증한다. 같은 정밀도(F32)로
비교했을 때 `bge-small`은 최소 0.999999, `harrier-270m`은 132문장 전부에서
1.000000이다. 후자의 코퍼스에는 한국어·일본어·중국어·키릴·아랍어·이모지가 포함된다.

양자화된 GGUF도 그대로 읽는다. `harrier-270m`을 같은 기준값과 비교하면
q8_0은 평균 0.999754(최소 0.999117), q4_k는 평균 0.984729(최소 0.946921)다.
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
테스트를 실행하려면 세 파일을 모두 둔다.

```text
models/bge-small-en-v1.5-f16.gguf
models/harrier-270m.gguf
models/harrier-270m-q8_0.gguf
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
