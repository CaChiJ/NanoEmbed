# 홈서버 인수인계: M5 배치 성능 측정 이어서 하기

작성일: 2026-08-30
브랜치: `feat/m4-tokenizer-memory` (커밋 `0804f1c`)

이 문서는 두 부분이다. 1절은 사람이 실행할 셋업 명령이고, 2절은 작업을 이어받을
AI 에이전트에게 그대로 붙여넣을 프롬프트다.

---

## 1. 홈서버 셋업

### 1.1 클론

서브모듈(`third_party/ggml`)이 있으므로 `--recursive`가 필요하다.

```sh
git clone --recursive -b feat/m4-tokenizer-memory \
  https://github.com/CaChiJ/NanoEmbed.git
cd NanoEmbed

# 이미 클론했다면
git submodule update --init --recursive
```

### 1.2 모델 내려받기

모델은 저장소에 없다. 네 개 중 앞의 셋은 테스트에 필수이고, Q4는 report-only다.

```sh
mkdir -p models
curl -fsSL -o models/bge-small-en-v1.5-f16.gguf \
  https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f16.gguf
curl -fsSL -o models/harrier-270m.gguf \
  https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m.gguf
curl -fsSL -o models/harrier-270m-q8_0.gguf \
  https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m-q8_0.gguf

# 선택 (report-only 정확도 측정용)
curl -fsSL -o models/harrier-270m-q4_k.gguf \
  https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m-q4_k.gguf
```

CMake가 `models/` 아래 이 이름들을 자동으로 찾는다. 다른 곳에 두려면
`-DNANOEMBED_TEST_MODEL=...` 등으로 지정한다.

### 1.3 Python 도구

벤치 러너가 `pyyaml`을 쓴다. 나머지는 fixture/golden 재생성용이라 측정만 할
거라면 `pyyaml`만 있어도 된다.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
# 최소 설치로 충분한 경우
# .venv/bin/pip install pyyaml
```

### 1.4 빌드

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
```

### 1.5 확인

```sh
ctest --test-dir build --output-on-failure
```

36개 전부 통과해야 한다. 통과하지 않으면 측정에 들어가지 말고 원인부터 볼 것.

**중요**: `nanoembed-bench`는 `/proc` 기반이라 **Linux에서만 빌드된다.**
macOS에서는 이 타깃이 아예 만들어지지 않으므로 측정을 할 수 없다.

### 1.6 측정 환경 조건

이전 측정이 실패한 이유가 환경 불안정이었다. 홈서버에서는 다음을 확인할 것.

- 벤치 실행 중 **다른 작업을 돌리지 말 것**. 이전 측정에서 배치 2가 실행마다
  37.9 / 25.0 / 42.3으로 흔들렸다.
- 메모리 여유를 확인할 것. 배치 128 측정이 Docker 데몬을 멈춘 전례가 있다.
- 각 측정은 **3회 이상** 돌리고 중앙값을 쓸 것.

---

## 2. 이어받을 AI 에이전트용 프롬프트

아래 전체를 그대로 붙여넣으면 된다.

---

### 배경

NanoEmbed는 GGUF 임베딩 모델을 C++/ggml로 실행하는 라이브러리다. 모델 가중치를
전부 메모리에 올리지 않고 레이어별로 mmap 페이지를 상주시키는 스트리밍 모드가
있다(M4, Linux 전용).

M5는 "한 레이어를 읽은 김에 여러 문장을 같이 계산한다"는 배치 기능이다. 기능과
정확도는 구현됐지만 성능이 문제였고, 원인을 추적해 해결한 상태다. 경과는
`docs/m5-overview.ko.md`와 `docs/m5-padding-removal.ko.md`에 있다. 먼저 읽을 것.

### 지금까지 확정된 사실

1. **배치가 느려진 원인은 패딩이다.** 길이가 다른 문장을 한 직사각형에 담으면
   빈칸이 생기고, 그 빈칸도 전부 계산된다. 길이가 균일한 대조 코퍼스
   (`uniform_len`)에서는 같은 조건에서 배치가 이득이었다.
2. **ggml은 마스크된 칸을 건너뛰지 않는다.** 점수판은 마스크 적용 전에 이미
   만들어지고 softmax도 패딩 폭까지 훑는다. 마스크는 정답을 맞추는 장치일 뿐
   계산을 줄이지 않는다. 미리 잘라내야 실제로 계산을 안 한다.
3. **문장별 어텐션이 정확도 문제를 해결했다.** 마스크 경로에서 harrier의 최대
   오차가 4.36e-5였는데, 문장별로 실제 길이만 계산하니 순차 처리와 **정확히
   동일**해졌다(오차 0). 원래 목표였던 1e-5를 통과한다.

### 현재 구현 상태

세 가지 어텐션 조립 방식이 있고, `src/arch/gemma3_arch.h`와
`src/arch/gemma3_arch.cpp` 두 파일의 같은 플래그로 전환한다.

| 방식 | `seq_lengths` | `packed` | 설명 |
|---|---|---|---|
| ① 마스크 | false | false | 원래 방식. 패딩 계산 + 마스크 |
| ② 문장별 어텐션 | true | false | 어텐션만 문장별. 나머지는 패딩 계산 |
| ③ 패킹 | true | true | **현재 기본값.** 모든 토큰 단위 연산에서 패딩 제거 |

③이 현재 체크아웃된 상태다. 조립은 `ggml_set_inplace`로 각 조각을 목적지에
한 번씩만 쓴다(`src/forward/gqa_attention.cpp`).

BERT는 아직 ①만 지원한다. gemma3(harrier)만 ②③을 쓴다.

### 해야 할 일

**세 방식의 성능을 같은 환경에서 측정하고 비교하라.**

시나리오는 `bench/m5-variant-sweep.yaml`에 있다. 배치 1·2·5·10·16·32·64,
harrier Q8, mixed_short 132문장, streaming, layer 파티션, warm 캐시다.

```sh
# 방식 전환: 두 파일에서 /*seq_lengths=*/X, /*packed=*/Y 를 바꾼다
# 예) ① 마스크로: 둘 다 false
# 그리고 재빌드 후 실행
cmake --build build -j "$(nproc)"

python3 bench/runner.py \
  --milestone M5-VARIANT \
  --scenarios bench/m5-variant-sweep.yaml \
  --bench build/bin/nanoembed-bench \
  --out bench/results/<환경이름>/variant-sweep/mask-run1.json
```

각 방식마다 **3회** 돌리고 중앙값을 쓴다.

### 측정 규칙 — 반드시 지킬 것

1. **배치 1은 대조군이다.** 세 방식 모두 배치 1에서는 코드 경로가 같다. 세 값이
   서로 크게 다르면 환경이 흔들린 것이므로 그 측정은 버린다.
2. **단일 실행을 믿지 말 것.** 이전 측정에서 첫 실행이 체계적으로 8% 느렸고,
   다른 작업과 겹친 실행은 배치 1이 41.4 대신 12.6으로 나왔다.
3. **이전 환경의 수치와 비교하지 말 것.** 이 저장소에 있는
   `bench/results/M5-docker-desktop-arm64/` 아래 수치는 전부 다른 머신(Docker
   Desktop arm64, 이후 재시작됨)의 것이다. 참고만 하고 판정 근거로 쓰지 말 것.
   새 환경에서는 새 기준선을 잡는다.
4. **결과를 낼 때는 실행 간 편차를 함께 보고할 것.** 차이가 편차보다 작으면
   "구분할 수 없다"고 쓸 것. 없는 결론을 만들지 말 것.

### 열린 질문

측정으로 답해야 할 것들이다.

1. **배치를 어디까지 키우는 게 이득인가?** 이전 환경에서는 배치 5쯤에서 이득이
   포화했다. 그 너머에서 꺾이는지, 꺾인다면 어디서인지 모른다.
2. **③의 제자리 쓰기가 실제로 효과가 있는가?** ①→②→③으로 오면서 조립 방식을
   두 번 바꿨는데(왼쪽 접기 → 짝 접기 → 제자리 쓰기), **마지막 두 변경은 성능을
   측정하지 못했다.** 정확성만 확인된 상태다. 커밋 `5a77500`, `198ad5d` 참고.
3. **배치 128이 왜 환경을 멈추게 했는가?** 이전 머신에서 배치 128을 포함한
   측정이 Docker 데몬을 멈췄다. 제자리 쓰기로 바꾼 지금도 그런지 확인이 필요하다.
   64가 무사히 끝난 뒤에 시도할 것.
4. **cold 캐시에서도 같은 순서인가?** cold 측정은
   `--cache-state cold --strict-cold`를 붙이고 `cache_state: cold` 시나리오를
   쓴다. cold는 편차가 크므로(이전 환경에서 중앙값 20%) 큰 차이만 판정할 것.

### 그 다음 후보

측정이 끝난 뒤 판단할 것.

- **BERT에 ②③ 적용.** 지금은 gemma3만 쓴다. BERT는 최대 오차 2.345e-4가 그대로고
  성능 이득도 못 받는다. `InputRequirements`에 플래그를 켜고 `attention.cpp`에
  같은 구조를 넣으면 된다.
- **묶음별 토큰·패딩 수를 벤치 결과로 노출.** `InternalStreamingDiagnostics`에
  `valid_tokens_processed`, `padding_tokens_processed`가 있지만 결과 JSON에
  나오지 않는다. 이 값 없이는 패킹의 이득이 입력 분포에 따라 어떻게 달라지는지
  정량화할 수 없다.
- **동시 요청 측정.** 지금까지 모든 측정은 요청을 하나씩 순서대로 처리한 것이다.
  서버 조건은 재본 적이 없고, 벤치 도구에 동시 실행 기능이 없다.

### 프로젝트 규칙

- **숨은 자동 동작을 넣지 말 것.** `max_batch`가 유일한 분할 기준이고, 환경변수
  스위치나 자동 fallback은 넣지 않는다. 측정이 인자만으로 재현 가능해야 한다.
- **문서는 한국어로, 쉬운 말로 쓸 것.** 영어 개념을 직역한 한자어를 쓰지 말고
  동작으로 풀어 쓸 것. 예: "상각한다"(X) → "여러 문장이 나눠 낸다"(O).
- **측정하지 않은 것을 측정한 것처럼 쓰지 말 것.** 커밋 메시지와 문서에 무엇이
  측정됐고 무엇이 안 됐는지 명시할 것. 기존 커밋들이 그 예시다.
