# 가중치 스트리밍과 메모리 결과 읽기

이 문서는 NanoEmbed의 레이어/가중치 스트리밍 결과를 읽기 위해 필요한
메모리·GGML 개념을 설명한다. 대상 독자는 mmap, PSS, graph allocator에 익숙하지
않은 개발자다. 결론부터 말하면, 현재 최적화는 **모델 가중치가 차지하는
file-backed 메모리**를 줄인다. 그러나 tokenizer/모델 메타데이터와 추론 계산용
버퍼가 차지하는 **anonymous(익명) 메모리**는 거의 건드리지 않는다.

여기서 쓰는 수치는 `harrier-270m F32`, warm cache, profile-on, 짧은 영어 문장
100개를 반복한 추가 측정값이다. 원래 M4 매트릭스와는 컨테이너와 워크로드가
다르므로 절대값을 서로 섞어 비교하지 않는다. 자세한 원본 결과는
[`CLOSEOUT.md`](../bench/results/M4-docker-desktop-arm64/CLOSEOUT.md)의
“Measured RSS/PSS/USS peaks” 절을 본다.

## 한 문장으로 보는 결과

가중치를 더 작은 단위로 나누자, 관측된 file-backed PSS peak는
27.50 MiB에서 17.83 MiB까지 줄었다. 반면 계산 버퍼와 중간값이 들어가는
anonymous PSS peak는 75.85 MiB로 그대로여서, 전체 PSS는 103.34 MiB에서
93.68 MiB까지만 줄었다.

| preset | 전체 PSS peak | file-backed PSS peak | anonymous PSS peak |
|---|---:|---:|---:|
| `layer` | 103.34 MiB | 27.50 MiB | 75.85 MiB |
| `budget:10MiB` | 93.68 MiB | 17.83 MiB | 75.85 MiB |
| `unit` | 94.32 MiB | 18.47 MiB | 75.85 MiB |

`budget:10MiB` 행만 보면 대략 다음과 같다.

```text
layer        : 가중치 파일 페이지 27.50 MiB + 익명 계산 메모리 75.85 MiB
budget:10MiB : 가중치 파일 페이지 17.83 MiB + 익명 계산 메모리 75.85 MiB
```

따라서 가중치 쪽은 9.67 MiB 줄었지만, 전체 프로세스 관점에서는 고정된
75.85 MiB가 남아 있으므로 전체 PSS 감소폭은 약 9.7 MiB, 즉 9.3%다.

## 먼저 알아둘 세 가지: 파일, 페이지, 프로세스 메모리

### GGUF 가중치 파일

모델의 수억 개 숫자(가중치)는 GGUF 파일에 들어 있다. Harrier F32 파일은 매우
크지만, 추론의 한 순간에는 모든 레이어 가중치를 동시에 읽을 필요가 없다.

### mmap: 파일을 “메모리처럼 보이게” 연결하는 방법

`mmap`은 파일 전체를 곧바로 RAM에 복사하는 기능이 아니다. 운영체제에게
“이 파일의 이 구간을 이 프로세스의 주소 공간에서 배열처럼 접근하게 해 달라”고
요청하는 기능이다.

실제 RAM은 보통 4 KiB 정도의 **페이지(page)** 단위로 필요해질 때 채워진다.
가중치 주소를 처음 읽으면 해당 페이지가 RAM에 올라오고(page fault), 더 이상
필요 없다고 알려 주면 커널은 나중에 그 페이지를 회수할 수 있다.

```text
GGUF 파일 ── mmap ──> 프로세스 주소 공간
                         └─ 실제로 읽은 페이지들만 RAM에 resident
```

NanoEmbed의 스트리밍은 각 레이어 group을 계산하기 직전에 필요한 GGUF 범위를
`madvise(WILLNEED)`로 요청하고, 계산이 끝나면 `madvise(DONTNEED)`로 회수 후보로
만든다. `madvise`는 요청(advice)이지 강제 삭제 명령은 아니다.

### anonymous memory: 파일 원본이 없는 메모리

`malloc`, `new`, thread stack, CPU backend의 계산 버퍼처럼 파일에 대응하지 않는
메모리를 Linux는 anonymous memory라고 부른다. 이 메모리는 어떤 GGUF 파일을
내린다고 자동으로 줄지 않는다.

```text
file-backed memory                 anonymous memory
------------------                 ----------------
GGUF mmap 가중치 페이지             activation(중간 계산값)
공유 라이브러리 코드/데이터          ggml 계산 버퍼
                                     tokenizer의 lookup/merge table
                                     GGUF metadata, C++ heap, thread stack
```

## RSS, PSS, USS는 무엇이 다른가

운영체제는 같은 파일 페이지를 여러 프로세스가 공유할 수 있다. 그래서 “프로세스가
차지한 메모리”에는 서로 다른 질문에 답하는 세 지표가 있다.

- **RSS (Resident Set Size)**: 지금 이 프로세스 주소 공간에서 실제 RAM에 올라온
  페이지의 합이다. 공유한 파일 페이지도 이 프로세스 입장에서는 전부 센다.
- **PSS (Proportional Set Size)**: 공유 페이지는 공유한 프로세스 수로 나누어 센다.
  예를 들어 4 MiB 파일 페이지를 두 프로세스가 공유하면 각 프로세스의 PSS에는
  2 MiB씩 들어간다. 여러 프로세스가 공존하는 환경의 실제 부담을 보기에 좋다.
- **USS (Unique Set Size)**: 다른 프로세스와 공유하지 않는 private page만 센다.
  프로세스를 종료했을 때 다른 프로세스에 영향 없이 사라질 가능성이 큰 메모리다.

이 벤치마크는 `/proc/<pid>/smaps_rollup`의 `Pss_Anon`과 `Pss_File`도 읽는다.

- **PSS anon**: anonymous page에 해당하는 PSS
- **PSS file-backed**: 파일 매핑 page에 해당하는 PSS. 여기서는 주로 mmap된 GGUF
  가중치다. 공유 라이브러리 등의 작은 항목도 포함될 수 있으므로 “GGUF만 정확히
  100%”라는 뜻은 아니다.

샘플은 10 ms마다 찍는다. 따라서 “peak”는 그 사이를 놓칠 수 있는 sampled peak이며,
커널이 제공하는 절대 high-water mark가 아니다. 또한 각 열은 각각의 최대 샘플을
요약한 값이므로, 반올림이나 peak 시점 차이 때문에 표의 component 합이 total PSS와
몇 KiB 정도 정확히 일치하지 않을 수 있다.

## GGML graph란 무엇인가

한 번의 추론은 행렬 곱, 정규화, softmax, residual add처럼 많은 연산의 연결이다.
GGML은 이를 실행 전에 **graph**로 표현한다.

```text
입력 x ──> Q/K/V 투영 ──> attention ──> residual add ──> FFN ──> 다음 레이어 x
              node들            node들          node들        node들
```

graph에는 다음이 포함된다.

- 어떤 tensor가 입력·출력·중간값인지
- 어떤 연산(node)이 어떤 tensor를 읽는지
- 연산의 순서와 의존성

현재 `layer` preset은 한 transformer block의 attention과 FFN을 큰 graph 하나로
만든다. `attn-ffn`, `budget`, `unit` preset은 이것을 여러 graph로 나눈다.

```text
layer:  [ attention + FFN ]

unit:   [norm] [QKV] [attention core] [post-attention]
        [FFN norm] [gate+up] [down] [post-FFN]
```

여기서 “unit”은 GGUF tensor 하나마다 무조건 graph 하나라는 뜻은 아니다. 예를
들어 Harrier의 `ffn_gate`와 `ffn_up`은 GeGLU 융합 연산에서 함께 필요하므로 하나의
실행 unit으로 남는다. 이 둘의 합이 10,240 KiB라서 더 나눈 경우에도 한 group의
가중치 peak는 10 MiB 아래로 내려가지 않는다.

## activation과 activation slot

**activation**은 모델 가중치가 아니라, 입력 문장을 계산하는 동안 생기는 중간 결과다.
예를 들어 Q, K, V, attention 결과, FFN의 hidden 결과가 activation이다. 입력 길이가
길수록 보통 커진다.

하나의 큰 graph 안에서는 GGML이 activation의 수명을 보고 같은 계산 버퍼 영역을
재사용할 수 있다. 하지만 graph를 둘로 나누면 앞 graph의 출력이 뒤 graph의 입력이
되어야 한다.

```text
앞 group graph                       다음 group graph

activation A ── download ──> slot ── upload ──> activation A
```

여기서 **activation slot**은 graph 경계를 넘겨야 하는 activation을 CPU heap에 잠시
보관하는 장소다. 현재 구현의 `SlotStore`가 이 역할을 한다.

따라서 group을 더 잘게 나누면 가중치 페이지는 줄일 수 있지만, slot은 오히려 늘 수
있다. Harrier F32의 진단값은 `layer` 77 KiB에서 `unit` 899 KiB로 증가했다. 다만
899 KiB는 75.85 MiB 규모의 전체 anonymous 메모리 중 작은 일부다.

## ggml_gallocr는 무엇인가

`ggml_gallocr`는 graph allocator의 약자다. graph를 실행하기 전에 그 graph가 만들
activation들의 shape와 수명을 보고, 필요한 계산 메모리(buffer)를 계획하고 확보한다.

중요한 점은 다음과 같다.

- 가중치 GGUF 파일을 저장하는 도구가 아니다.
- activation과 임시 계산 결과를 위한 CPU backend buffer를 관리한다.
- 현재 streaming context 하나가 하나의 `ggml_gallocr`를 갖고, 여러 group graph가
  그것을 계속 공유한다.
- 한 번 필요한 큰 compute buffer를 확보하면 context가 살아 있는 동안 allocator가
  그 용량을 보유할 수 있다. 작은 다음 graph를 실행한다고 즉시 작아지지 않는다.

그래서 레이어 group을 쪼개도 **가장 큰 group이 요구하는 activation scratch** 또는
embedding/final phase가 요구하는 scratch가 같다면 anonymous peak는 그대로일 수
있다. 다만 Harrier의 75.85 MiB를 전부 `ggml_gallocr`가 쓴다고 단정할 수는 없다.
`smaps_rollup`은 heap 전체만 보여 줄 뿐 heap 안의 C++ 객체별 소유자를 알려 주지
않는다.

이것은 메모리 누수라는 뜻이 아니다. 반복 추론에서 매번 재할당하지 않도록 하는
의도적인 context 재사용 전략이다. 다만 “전체 프로세스 peak를 더 낮추기”라는
목표에는 별도 최적화 대상이 된다.

관련 구현 위치:

- streaming context와 `ggml_gallocr` 생성:
  [`src/streaming_execution.cpp`](../src/streaming_execution.cpp)
- graph 경계 slot 저장소:
  [`src/streaming_execution.cpp`](../src/streaming_execution.cpp)
- `Pss_Anon`/`Pss_File` 파싱:
  [`tools/nanoembed-bench/metrics.cpp`](../tools/nanoembed-bench/metrics.cpp)

## 현재 결과를 단계별로 해석하기

### 1. 가중치 lease의 정적 상한은 실제로 줄었다

Harrier F32의 한 block 전체는 21,772 KiB다. `layer`는 이를 한 번에 lease한다.
`budget:10MiB`와 `unit`은 가장 큰 단위가 10,240 KiB이므로 한 번에 유지하도록
요청하는 레이어 가중치 상한이 약 53% 낮다.

이 값은 **코드가 요청한 범위**이지 RSS가 아니다.

### 2. 실제 file-backed PSS도 감소했다

`layer`의 file-backed PSS peak 27.50 MiB가 `budget:10MiB`에서 17.83 MiB로
내려갔다. 실제 resident GGUF page가 줄었다는 방향성 증거다.

정적 lease 상한은 53% 감소했는데 관측 file-backed PSS는 35.2%만 감소했다. 이는
정상적인 차이다.

- `madvise(DONTNEED)`는 즉시·강제 회수가 아니다.
- final norm처럼 수명 전체에 유지하는 common tensor가 있다.
- mmap과 `madvise`는 page 단위로 반올림된다.
- 10 ms 샘플은 정확히 같은 계산 시점을 포착하지 않는다.

### 3. 당시 측정에서 PSS anon은 partitioning만으로 줄지 않았다

아래 75.85 MiB 수치는 tokenizer 메모리 최적화 전의 B1 측정값이다. 당시에는
PSS anon이 모든 preset에서 baseline, peak, final까지 거의 같았다.
partitioning은 GGUF weight lease만 바꾸며, 다음 익명 메모리의 크기는 바꾸지 않는다.

- `ggml_gallocr`의 compute/activation buffer
- graph 경계를 넘기는 slot buffer
- Harrier tokenizer의 BPE character lookup table과 디스크 merge index fence
- 그 시점에 GGUF가 계속 보관하던 tokenizer metadata heap

특히 Harrier는 262,144개 vocab과 514,906개 merge rule을 가진다. tokenizer는
현재 merge rule을 4KiB page 기반 로컬 캐시에 보관하고 작은 fence만 RAM에 유지하며,
tokenizer 생성 뒤 소비한 큰 GGUF metadata 배열도 제거한다. 따라서 위의 절대 PSS anon
수치는 현재 구현의 steady-state 기대값이 아니다. 다만 layer partition 자체가 activation,
character lookup, tokenizer fence를 줄이지 않는다는 해석은 그대로 유효하다.

### 4. 그래서 전체 PSS/RSS의 변화가 작아 보인다

전체 메모리에서 anonymous compute 영역이 더 크면, 가중치 영역을 반으로 줄여도
합계는 조금만 줄어든다. 이번 Harrier F32에서는 file-backed peak보다 anonymous
peak가 약 세 배다.

## `unit`이 `budget:10MiB`보다 더 좋지 않은 이유

두 preset 모두 최대 group weight가 10,240 KiB라는 같은 하한에 도달한다. 하지만
`budget:10MiB`는 레이어당 3 graph이고 `unit`은 8 graph다.

`unit`은 추가 graph 경계, slot upload/download, `madvise` 호출, allocator replan을
만든다. 그 결과 이번 한 번의 profile-on 측정에서 `unit`의 file-backed PSS peak는
18.47 MiB로 `budget:10MiB`의 17.83 MiB보다 약간 높고, 처리량도 더 낮았다.

이는 “unit이 가중치 상한을 못 낮췄다”가 아니라, 이미 같은 최대 tensor 하한에
도달한 뒤에는 더 잘게 쪼개는 비용만 추가될 수 있다는 뜻이다. memory profile은
한 번의 관측이며 10 ms sampling이므로, 이 0.64 MiB 차이 자체를 정밀한 우열로
해석해서는 안 된다.

## PSS anon까지 줄이려면 무엇이 달라져야 하나

가중치 streaming을 더 잘게 쪼개는 것만으로는 부족하다. 익명 메모리를 줄이는
별도 작업이 필요하다. 우선 `ggml_gallocr_get_buffer_size()`와 tokenizer/metadata
생성 전후의 heap 계측을 노출해 어느 쪽이 얼마나 차지하는지 확정해야 한다. 그 뒤의
후보는 다음과 같다.

1. graph별 activation lifetime을 다시 설계해 `ggml_gallocr`가 필요로 하는 최대
   compute buffer를 낮춘다.
2. Harrier BPE merge table을 더 조밀한 자료구조로 바꾸고, tokenizer 생성 후 더 이상
   필요 없는 GGUF metadata를 해제할 수 있는지 수명 계약을 검토한다.
3. context 길이와 activation shape에 맞는 scratch reservation 전략을 검토한다.
   단, 단순히 작게 잡으면 긴 입력에서 재할당·실패·성능 저하가 생길 수 있다.
4. graph 경계 activation을 더 작게 저장하거나, 불필요한 slot을 없앤다. 다만
   현재 slot은 전체 anon의 작은 부분이므로 이것만으로 큰 절감은 기대하기 어렵다.
5. group마다 allocator를 새로 만드는 방식은 peak를 낮출 가능성이 있지만, 반복
   할당과 재계획 비용이 커질 수 있다. 현재 `unit`이 이미 latency를 악화시킨
   결과를 고려해, 반드시 profile-off 성능과 정확도를 함께 다시 측정해야 한다.

따라서 이번 결과가 말하는 바는 명확하다.

> 현재 최적화는 “가중치 mmap 페이지를 덜 상주시킨다”에는 성공했다. “추론 전체의
> activation/compute 메모리를 줄인다”는 아직 별도의 문제다.
