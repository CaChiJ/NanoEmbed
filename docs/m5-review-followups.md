# M5 코드 리뷰 후속 결정 기록

작성일: 2026-08-28

이 문서는 M5 구현 리뷰에서 확인했지만, 공개 동작·정확도 계약·CI 비용 또는 다음
최적화 방향에 대한 사용자 결정을 수반하므로 자동으로 변경하지 않은 항목을 보존한다.
M5는 기능 구현 상태다. 전체 요약은 [M5 정리](m5-overview.ko.md)에 있다.

> **2026-08-28 갱신**: 최초의 "수용 판정 실패"는 파일이 이미 메모리에 있고
> 입력 길이 편차가 큰 조건 하나에서 나온 것이다. 길이가 고른 입력에서는 +56%,
> 디스크에서 읽는 조건에서는 +356~523%로 기준을 넘는다. 원인은 구현이 아니라
> 패딩으로 확정됐다. 아래 표의 우선순위는 이 결과를 반영해 다시 볼 필요가 있다.

## 즉시 수정한 결함

Streaming embedding/group 경로가 residency lease를 획득한 뒤 graph metadata context를
생성하고 있었다. context 생성이 실패하면 lease destructor가 `compute_complete=false`로
release를 시도해 coordinator를 poison하고 active lease를 남길 수 있었다.

- graph context와 token-range 준비가 성공한 뒤에만 lease를 획득하도록 순서를 변경했다.
- lease 획득 이후의 동기 실패는 safe-to-release로 표시한 뒤 scope 종료 시 정리한다.
- embedding context 실패와 첫 group context 실패를 각각 주입한다.
- 실패 시 batch output 전체 NaN, active lease 0, coordinator 비-poison을 검사한다.
- 같은 streaming context를 실패 직후 재사용해 eager batch와 parity가 유지되는지 검사한다.

공개 C ABI와 정상 실행 경로는 변경하지 않았다.

## 사용자 결정이 필요한 항목

| 우선순위 | 항목 | 현재 근거 | 권장안 | 완료 조건 |
|---|---|---|---|---|
| 높음 | 실제 batch를 기본 경로로 배포할지 | Harrier Q8 streaming B32가 control 34.357보다 낮은 16.302 items/s | 성능 gate를 넘기 전에는 M5를 merge-ready/default로 간주하지 않는다. 롤백보다는 별도 최적화 브랜치에서 token-budget/length-aware subdivision을 먼저 측정한다. | B32 item p50와 items/s 모두 control 대비 15% 이상 개선, B128 throughput 개선 |
| 높음 | 원 정확도 계약 유지 여부 | BERT F16 max abs 2.345e-4로 원 기준 1e-5 실패 | 관측값에 맞춘 완화 gate를 원 계약의 대체로 승인하지 않는다. padded kernel parity를 고치거나, 허용오차 변경을 명시적으로 승인한다. | 승인된 동일 기준으로 macOS/Linux BERT·Harrier 모두 통과 |
| 높음 | padding 계산량과 M6 순서 | 정확히 B=max_batch인 canonical workload에서는 length sort가 한 dense batch의 S를 줄이지 못함 | activation 압축만 진행하기 전에 token-budget 또는 길이 편차 기반 subdivision을 독립 실험한다. | padding ratio, dense attention FLOP 추정, throughput/RSS를 같은 corpus로 비교 |
| 중간 | 전체 allocation-failure 계약 | 이번 수정은 graph context 경계만 deterministic하게 검사함 | gallocr allocation, host staging allocation, backend compute 실패를 주입하고 C API status/output 계약을 검사한다. | OOM→`NANOEMBED_ERR_OOM`, 실행 실패→전체 NaN, 이후 context 상태가 문서 계약과 일치 |
| 중간 | 10,000회 안정성 soak | 아직 미실행 | 고정 shape eager/streaming을 별도 장시간 job으로 실행하고 RSS/FD/VM-region slope를 저장한다. | 지속 증가 없음과 재현 가능한 raw timeline |
| 중간 | Q4 회귀 gate | 현재 Q4는 report-only이고 CI 필수 다운로드 대상이 아님 | 지원 포맷으로 보장한다면 Q4 모델을 CI cache와 정확도 gate에 포함한다. | 모델 부재 시 CI 실패, batch/eager/streaming parity gate 통과 |
| 중간 | benchmark source provenance | 저장 결과의 `git_sha`와 code fingerprint가 unavailable | host에서 commit, dirty 여부, binary diff digest를 수집해 result에 넣는다. | 결과 파일만으로 source→binary 관계를 감사 가능 |
| 낮음 | p99 표본 수 | 독립 실행당 timed batch가 B32 5개, B128 3개 | p99를 descriptive maximum에 가깝다고 표시하거나 timed batch를 100개 이상으로 늘린다. | percentile population과 표본 수를 CLOSEOUT에 명시 |

## 변경하지 않은 이유

- 실제 batch를 sequential control로 되돌리면 사용자가 요청한 M5 기능 자체를 제거한다.
- 환경변수나 숨은 자동 fallback을 넣으면 `max_batch`가 유일한 분할 기준이라는 현재 계약을
  우회한다.
- 정확도 gate 완화는 기술 수정이 아니라 호환성 정책 변경이다.
- Q4 필수 CI와 10,000회 soak는 다운로드·실행 비용 및 CI 구조 선택이 필요하다.

따라서 위 항목은 기록만 남기며, 명시적 결정 전까지 M5 성공 또는 기본경로 승인으로
해석하지 않는다.
