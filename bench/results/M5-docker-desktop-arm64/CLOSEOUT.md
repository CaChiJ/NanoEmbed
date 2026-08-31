# M5 실제 레이어 단위 배치 closeout

## 결론

M5의 실제 B축 기능 구현과 측정은 끝났지만, **warm 캐시 기준 M5 수용 판정은 실패**다.

> **2026-08-28 추가**: 이 문서의 성능 판정은 전부 warm 캐시 측정이다. warm에서는
> 문장 하나당 major fault가 0이어서, 배치가 줄이려던 디스크 읽기가 존재하지 않았다.
> 같은 구현을 cold 캐시에서 재측정하면 배치 1 → 10에서 처리량이 356~523% 오르고
> 문장당 디스크 읽기가 정확히 1/B로 준다. 또한 warm이라도 입력 길이가 균일하면
> 같은 배치 10이 +56%다. 즉 gate를 못 넘는 조건은 **warm + 길이 편차 큼** 하나이고,
> 원인은 입력 길이 편차에 따른 패딩으로 확정됐고 이후 Harrier의 패딩 계산을
> 제거했다. 초기·최종 수치와 방법은
> [`docs/m5-overview.ko.md`](../../../docs/m5-overview.ko.md)에 통합돼 있다.

- eager와 Linux streaming 모두 실제 `[H,S,B]` graph를 실행한다.
- stable length sort, `max_batch` 분할, right padding, attention/pooling mask,
  원래 순서 복원과 batch 단위 streaming lease가 동작한다.
- 공개 C ABI와 `nanoembed_context_params` layout은 바뀌지 않았다.
- 기능·동시성·streaming phase-count는 통과했다.
- 원래 정확도 gate와 batch 32/128 성능 gate는 통과하지 못했다.

수치를 맞추기 위한 숨은 소배치, token packing, eager fallback 또는 자동 재시도는
추가하지 않았다.

## 구현 경계

- `src/batch.{h,cpp}`: 전체 토큰화, stable sort, sub-batch와 mask materialization
- `src/embedder.cpp`: eager true batch와 단일 B=1 무-mask 경로
- `src/streaming_execution.cpp`: `[H,S,B]` SlotStore, sub-batch당 embedding/layer/final 1회
- `src/forward/pool.cpp`: masked Mean과 문장별 LAST gather
- `src/forward/linear.h`: shared 2-D weight projection을 `[S*B]` 한 GEMM으로 실행
- `src/api/c_api.cpp`: 선검증, 상태 코드 변환, 실행 실패 시 batch 전체 NaN
- benchmark schema v3: batch/item latency, batches/items per second, fault/read per item

## 정확도

원 계획은 cosine `>=0.999999`, max absolute `<=1e-5`였다. padded graph는 trimmed
B=1 graph보다 softmax row가 넓고, ARM vector reduction grouping이 달라 이 gate를
충족하지 못했다.

| 모델 | 최악 cosine | 최악 max abs | 판정 |
|---|---:|---:|---|
| BERT F16, macOS | 0.999999179 | 1.839e-4 | 원 gate 실패 |
| BERT F16, Linux arm64 boundary | 0.999998652 | 2.345e-4 | 원 gate 실패 |
| Harrier F32 | >=0.999999985 | 4.361e-5 | max-abs 원 gate 실패 |
| Harrier Q8_0 | 1.0 | 2.98e-8 | 통과 |
| Harrier Q4_K report-only | 1.0 | 2.98e-8 | 보고 전용 |

같은 길이만 묶은 mask-free batch는 모든 모델에서 sequential과 정확히 0 차이다.
회귀 테스트는 관측 근거에 따라 cosine `>=0.999998`, max absolute `<=2.5e-4`를
명시적으로 사용한다.

## Harrier Q8 streaming batch 32: 5회 독립 실행

profile-off 결과다. 각 native result 안에는 warmup 2회와 timed batch 5회가 있다.

| 경로 | batch p50 중앙값 ms | batch p90 중앙값 ms | item p50 중앙값 ms | items/s 중앙값 | lifetime RSS 중앙값 MiB |
|---|---:|---:|---:|---:|---:|
| M4식 sequential control | 798.071 | 845.023 | 24.940 | 34.357 | 55.020 |
| 실제 batch | 1,827.650 | 2,019.320 | 57.114 | 16.302 | 59.066 |

- item p50 변화: `+129.0%` (낮을수록 좋음)
- throughput 변화: `-52.6%` (높을수록 좋음)
- 요구 gate: 두 지표 모두 최소 15% 개선
- 판정: **실패**
- 10배 stretch 목표 달성률: 실제 speedup `0.474x / 10x = 4.7%`

선택된 32개 입력의 문자 길이는 11~148자로 넓고, 합 1,420자에 비해 한 dense
batch는 최대 길이 기준 4,736칸을 계산한다. token 수와 정확히 같지는 않지만 padding
증폭의 방향을 설명한다. shared linear GEMM flatten 최적화 후에도 dense attention과
activation의 padding 비용이 지배했다.

## Batch 128 및 eager 참고축

| 경로 | batch p50 ms | item p50 ms | items/s | lifetime RSS MiB |
|---|---:|---:|---:|---:|
| Harrier Q8 eager B128 | 7,804.39 | 60.972 | 16.776 | 434.473 |
| Harrier Q8 streaming B128 | 8,249.79 | 64.452 | 14.875 | 176.398 |
| Harrier Q8 streaming sequential control B128 | 3,281.43 | 25.636 | 37.562 | 55.043 |
| BERT F16 eager B32 | 4,342.38 | 135.699 | 7.314 | 96.723 |
| Harrier F32 eager B32 | 35,859.70 | 1,120.620 | 0.893 | 1,081.880 |

Batch 128 streaming도 sequential control보다 처리량이 높아야 한다는 gate를 실패했다.
대신 eager 대비 streaming RSS는 434.47→176.40MiB로 낮았다.

## Streaming preset별 batch 32 메모리

10ms profile-on 결과다. profile latency는 성능 표와 섞지 않는다. Peak는 RSS의
kernel lifetime/window peak와 sampled RSS/PSS/USS peak를 함께 기록했다.

| preset | RSS lifetime | RSS p50 | p90 | p95 | p99 | sampled peak |
|---|---:|---:|---:|---:|---:|---:|
| layer | 59.238 | 55.277 | 59.191 | 59.254 | 59.254 | 59.254 |
| attn-ffn | 59.902 | 57.512 | 60.113 | 60.176 | 60.176 | 60.176 |
| budget:10MiB | 58.965 | 55.004 | 58.918 | 58.981 | 58.981 | 58.981 |
| unit | 88.016 | 86.145 | 88.070 | 88.164 | 88.164 | 88.164 |

| preset | PSS p50 | p90 | p95 | p99 | peak | USS p50 | p90 | p95 | p99 | peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| layer | 53.033 | 56.947 | 57.010 | 57.010 | 57.010 | 51.625 | 55.539 | 55.602 | 55.602 | 55.602 |
| attn-ffn | 55.268 | 57.869 | 57.932 | 57.932 | 57.932 | 53.859 | 56.461 | 56.523 | 56.523 | 56.523 |
| budget:10MiB | 52.760 | 56.674 | 56.736 | 56.736 | 56.736 | 51.352 | 55.266 | 55.328 | 55.328 | 55.328 |
| unit | 83.900 | 85.826 | 85.920 | 85.920 | 85.920 | 82.492 | 84.418 | 84.512 | 84.512 | 84.512 |

unit의 context-local slot/allocator 비용이 가장 컸다. layer의 Pss_Anon은 46.852MiB,
unit은 76.313MiB로 profile 전체에서 고정적인 지배 항목이었다.

## Single-request 회귀

동일 `english_short:1` selection으로 M4 저장 결과와 비교했다.

| 경로 | M4 p50→M5 p50 ms | 변화 | M4→M5 lifetime RSS MiB | 판정 |
|---|---:|---:|---:|---|
| Harrier Q8 eager | 13.617→12.448 | -8.6% | 360.848→327.699 | 통과 |
| Harrier Q8 streaming | 17.482→18.352 | +5.0% | 87.035→55.000 | 통과 |

±15% single latency와 +5% RSS 허용 기준을 모두 만족했다.

## 테스트

- macOS Release: 36/36 pass, Linux-only 3개 명시적 skip
- Linux arm64: distinct-context true batch concurrency pass
- Linux arm64: streaming integration과 sub-batch phase diagnostics pass
- Linux arm64: benchmark selftest/profile selftest pass
- BERT/Harrier F32/Q8/Q4, Mean/CLS/LAST/default, equal/padded lengths 검증
- batch 크기 1, max_batch, max_batch+1, 2*max_batch+3 검증
- invalid middle null은 output untouched, 실행 실패 경로는 전체 NaN 계약
- Windows CI는 tokenizer와 synthetic batch planner target을 빌드/실행하도록 확장

## 수용 기준 판정

| 기준 | 결과 |
|---|---|
| 실제 eager/streaming B>1, output order, max_batch 분할 | 통과 |
| graph/lease 횟수가 item이 아니라 sub-batch에 비례 | 통과 |
| 원 계획 정확도 `0.999999 / 1e-5` | 실패 |
| batch 32 item p50와 items/s 각각 +15% | 실패 |
| batch 128가 sequential control보다 높은 처리량 | 실패 |
| single latency ±15%, single RSS +5% 이내 | 통과 |
| batch 32/128 절대 메모리 기록 | 통과 |
| fixed-shape 10,000회 RSS/FD/VM-region soak | 이번 bundle에서 미실행 |
| allocation failure injection | 측정 bundle에서는 미실행; post-review graph-context 경계 주입 테스트 추가 |
| 10배 stretch 목표 | 실패, 4.7% 달성 |
| Pss_Anon/slot activation이 M6 진입 근거인지 확인 | 확인 |

따라서 기능 코드는 M5 범위대로 구현됐지만 milestone을 성공으로 닫지는 않는다.
다음 실험은 plan에서 제외했던 token packing 또는 길이 편차를 고려한 명시적
sub-batch 정책, 그리고 M6 activation 압축을 각각 독립 축으로 측정해야 한다.

## Post-review correction

코드 리뷰에서 streaming graph metadata context가 residency lease 획득 뒤 생성되어,
context 생성 실패 시 active lease와 poisoned coordinator가 남을 수 있음을 확인했다.
2026-08-28에 context/range 준비 뒤 lease를 획득하도록 실패 경계를 수정하고 embedding과
첫 group context의 deterministic failure injection을 추가했다. 실패 output 전체 NaN,
active lease 0, 비-poison 상태와 같은 context의 후속 성공을 검사한다.

공개 동작 또는 정책 결정이 필요한 나머지 항목과 이후 처리 결과는
[`docs/m5-overview.ko.md`](../../../docs/m5-overview.ko.md)에 통합했다. 이 교정은
이 문서가 보존하는 당시 성능·정확도 측정 판정을 바꾸지 않는다.
