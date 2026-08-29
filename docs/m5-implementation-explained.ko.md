# M5 실제 레이어 단위 배치: 구현 해설

작성일: 2026-08-28

이 문서는 M5에서 추가된 "실제 배치(true batch)"가 어떻게 구현되었는지,
병렬 처리가 어느 층위에서 일어나는지, 레이어 사이의 중간 상태를 무엇이
들고 있는지를 설명한다. M5 전체 요약은 [M5 정리](m5-overview.ko.md)에 있다. 성능 수치와 수용 판정은
[`bench/results/M5-docker-desktop-arm64/CLOSEOUT.md`](../bench/results/M5-docker-desktop-arm64/CLOSEOUT.md)와
[배치×파티션 매트릭스](m5-batch-partition-matrix.ko.md)에 있다.

## 1. M5가 바꾼 것

M4까지 `nanoembed_embed_batch`는 이름만 배치였다. 내부에서는 문장 하나씩
`embed()`를 반복했다. M5는 같은 C ABI를 유지한 채 내부를 실제 `[H, S, B]`
그래프로 바꿨다.

```
M4:  for each text: [H,S,1] 그래프 한 번씩  →  가중치를 문장 수만큼 다시 읽음
M5:  문장 B개를 한 그래프에 쌓음 [H,S,B]   →  가중치를 sub-batch당 한 번만 읽음
```

기호는 `H` = 은닉 차원, `S` = 시퀀스 길이(패딩 후), `B` = sub-batch 안의 문장 수다.
ggml 텐서는 `ne[0]`이 가장 안쪽(연속) 축이므로 토큰 하나의 위치는 `s + S*b`로 계산된다.

핵심 동기는 스트리밍이다. 스트리밍 모드는 레이어를 바꿀 때마다 mmap된 파일
페이지를 다시 상주시킨다. 문장마다 전 레이어를 훑으면 그 I/O가 문장 수만큼
반복된다. B개를 한 번에 통과시키면 I/O 임대(lease) 횟수가 **문장 수가 아니라
sub-batch 수**에 비례한다.

## 2. 요청 하나가 지나가는 길

`nanoembed_embed_batch()` 호출부터 결과가 채워지기까지의 순서다.

```
nanoembed_embed_batch          src/api/c_api.cpp
  │  ① 선검증: null 포인터, n_texts<0, H*n_texts 오버플로
  │  ② const char* 를 std::string 으로 복사 (수명 소유권 확보)
  ▼
Embedder::embed_batch          src/embedder.cpp        (eager)
InternalStreamingModel::embed_batch  src/streaming_execution.cpp (streaming)
  │  ③ make_batch_plan: 전체 토큰화 + 길이 stable sort
  ▼
  for each sub-batch:
      ④ materialize_batch: 패딩, 마스크, 역-인덱스 생성
      ⑤ 그래프 구축 + 할당 + 입력 업로드
      ⑥ 실행
      ⑦ 결과를 원래 입력 순서로 scatter
```

③과 ④가 M5에서 새로 생긴 [`src/batch.h`](../src/batch.h) /
[`src/batch.cpp`](../src/batch.cpp)이고, 나머지는 기존 경로의 확장이다.

`batch.{h,cpp}`는 **요청에 종속된 데이터만** 소유한다. 토큰 ID, 길이 정렬,
패딩, 마스크가 전부다. 모델 가중치, 그래프 할당자, 활성값 슬롯은 원래 주인이
그대로 들고 있다. eager와 streaming 두 경로가 이 모듈 하나를 공유한다.

## 3. 배치 계획: 길이로 묶고, 나중에 되돌린다

### 3.1 왜 길이순으로 정렬하는가

한 그래프에 들어가는 문장들은 길이가 같아야 한다. 짧은 문장은 패딩 토큰으로
채워지는데, 그 패딩은 **계산은 되지만 버려지는** 값이다. 5글자 문장과
500글자 문장을 같이 묶으면 5글자짜리도 500칸을 계산한다.

그래서 `make_batch_plan`은 먼저 전부 토큰화한 뒤 토큰 길이로 **stable sort**
한다 ([`src/batch.cpp`](../src/batch.cpp)). 비슷한 길이끼리 이웃하게 되므로
sub-batch 안의 길이 편차가 줄어든다.

```
입력:      [A(50), B(3), C(48), D(5)]
정렬 후:   [B(3), D(5), A(50), C(48)]      ← 원래 인덱스를 함께 들고 다님
max_batch=2 로 분할:
  sub-batch 0: [B(3), D(5)]   → S=5,  패딩 2칸
  sub-batch 1: [A(50), C(48)] → S=50, 패딩 2칸
```

정렬하지 않았다면 `[A,B]`와 `[C,D]`가 묶여 각각 S=50, S=48이 되어
패딩이 47+43칸으로 폭증한다.

`stable_sort`를 쓴 이유는 같은 길이 문장들의 상대 순서를 보존하기 위해서다.
결과가 결정적이어야 같은 입력에 같은 sub-batch 구성이 나온다.

`BatchItem`은 `original_index`를 함께 들고 다닌다. 정렬은 계산 편의를 위한
내부 사정이므로, 호출자에게는 **입력 순서 그대로** 돌려줘야 한다.

### 3.2 sub-batch로 나누기

`max_batch`가 유일한 분할 기준이다.

```cpp
size_t BatchPlan::subbatch_count() const noexcept {
    return items.empty() ? 0 : 1 + (items.size() - 1) / max_batch;
}
```

토큰 예산이나 길이 편차에 따른 적응형 분할은 **일부러 넣지 않았다**. 숨은
분할 규칙이 있으면 `max_batch`가 계약이라는 말이 거짓이 되고, 성능 수치가
어떤 분할에서 나온 것인지 재현할 수 없게 된다.

여기서 나오는 부작용 하나: 마지막 sub-batch는 대개 덜 찬다. 132개 입력을
`max_batch=5`로 나누면 26개는 5개짜리, 마지막 1개는 2개짜리다. 그리고 정렬
때문에 **마지막 sub-batch가 가장 긴 문장들**을 담는다. 이것이 메모리 측정에서
배치 크기 대비 비단조적인 결과가 나오는 이유다.

### 3.3 materialize_batch: 패딩과 마스크

`materialize_batch`는 sub-batch 하나를 실제 텐서 내용으로 편다.
`S`는 그 sub-batch에서 **가장 긴** 문장의 길이 = 정렬됐으므로 마지막 원소의 길이다.

만드는 것은 다음과 같다.

| 필드 | 모양 | 용도 |
|---|---|---|
| `token_ids` | `[S,B]` I32 | 토큰 ID. 빈칸은 `padding_id` |
| `learned_positions` | `[S,B]` I32 | BERT 학습 위치 임베딩 인덱스 |
| `rope_positions` | `[S]` I32 | RoPE용 0..S-1. 문장 간 공유 |
| `type_ids` | `[S,B]` I32 | BERT segment ID, 전부 0 |
| `attention_mask` | `[S,S,1,B]` F16 | 패딩 키를 `-inf`로 |
| `valid_mask` | `[1,S,B]` F32 | Mean 풀링에서 패딩 제외 |
| `mean_scale` | `[1,B]` F32 | `1/실제길이` |
| `last_indices` | `[B]` I32 | LAST 풀링이 집을 평탄화 인덱스 |
| `original_indices` | `[B]` | scatter용 역-인덱스 |

**`padded` 플래그가 중요하다.** sub-batch 안의 모든 문장 길이가 같으면
(`plan.items[begin].ids.size() == S`) `padded=false`가 되고, 마스크 텐서를
아예 만들지 않는다. 마스크 없는 경로는 M4와 **비트 단위로 동일한** 그래프다.
이것이 정확도 회귀에서 "같은 길이만 묶은 배치는 sequential과 정확히 0 차이"가
나오는 이유다. 오차는 패딩된 그래프에서만 생긴다.

`rope_positions`만 `[S]`인 것도 의도적이다. ggml의 RoPE는 위치 벡터를 공유
벡터로 요구한다. 학습 위치 임베딩은 `[S,B]` 셀마다 인덱스가 필요하다. M4에서는
둘이 `pos_ids` 하나였는데, B=1일 때만 우연히 같은 모양이라 문제가 없었다.
M5는 `GraphInputs`에서 이 둘을 `learned_pos_ids` / `rope_pos_ids`로 분리해
B>1 그래프가 우연한 모양에 기대지 못하게 했다
([`src/arch/model_arch.h`](../src/arch/model_arch.h)).

### 3.4 오버플로를 조용히 넘기지 않는다

`S*B`, `S*S*B` 같은 곱은 전부 `checked_mul`을 통과한다. 실패하면
`AllocationError`를 던지고, C API에서 `NANOEMBED_ERR_OOM`으로 변환된다.
`std::bad_alloc`과 `std::length_error`도 같은 곳으로 모인다. 배치는 입력
개수가 곧 메모리이므로, 크기 계산이 조용히 감기면 곧바로 힙 오염이 된다.

## 4. 그래프에서 B축이 흐르는 방식

배치를 넣는다고 모든 연산이 저절로 B를 처리하지는 않는다. 세 군데를 손봤다.

### 4.1 공유 가중치 선형 변환: 하나의 큰 GEMM으로

`[H,S,B]` 활성값에 `ggml_mul_mat`을 그대로 쓰면 ggml은 B를 **분리된 행렬곱
B개**로 취급한다. 그런데 가중치 행렬은 모든 문장이 공유한다. 그래서 연속인
토큰 축과 배치 축을 평탄화해서 **한 번의 더 큰 GEMM**으로 만든다
([`src/forward/linear.h`](../src/forward/linear.h)).

```cpp
inline ggml_tensor * build_linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x) {
    if (x->ne[2] == 1 && x->ne[3] == 1) {
        return ggml_mul_mat(ctx, weight, x);      // B=1 그래프는 손대지 않음
    }
    const int64_t S = x->ne[1];
    const int64_t B = x->ne[2] * x->ne[3];
    ggml_tensor * flat = ggml_reshape_2d(ctx, x, x->ne[0], S * B);
    ggml_tensor * projected = ggml_mul_mat(ctx, weight, flat);
    return ggml_reshape_3d(ctx, projected, projected->ne[0], S, B);
}
```

reshape는 view이므로 복사가 없다. 토큰 축과 배치 축이 둘 다 연속이라 평탄화가
문장을 섞지 않는다. Q/K/V/O 선형 변환과 FFN이 모두 이 경로를 쓴다.

`B=1`을 분기로 빼둔 것은 회귀 방지다. 단일 문장 그래프는 M4와 글자 그대로
같은 노드를 만든다.

### 4.2 임베딩 테이블 조회

벤더된 ggml의 `GET_ROWS`는 소스와 인덱스의 배치 축이 맞아야 한다. 임베딩
테이블은 배치 축이 없는 공유 테이블이다. 그래서 `[S,B]` 인덱스를 `[S*B]`로
평탄화해 한 번 조회하고 `[H,S,B]`로 되돌린다
([`src/forward/embed_layer.cpp`](../src/forward/embed_layer.cpp),
[`src/arch/gemma3_arch.cpp`](../src/arch/gemma3_arch.cpp)). 양쪽 reshape 모두 view다.

### 4.3 패딩 인지 어텐션

패딩 토큰은 계산은 되지만 **다른 토큰이 그것을 쳐다보면 안 된다.** 어텐션은
모든 키를 훑기 때문에, 막지 않으면 패딩 위치의 쓰레기 값이 softmax를 통해
실제 토큰의 출력에 섞인다.

`attention_mask`는 `[S,S,1,B]` F16으로, 쿼리 `q`가 키 `k`를 볼 때
`k < 실제길이`면 `0`, 아니면 `-inf`를 담는다. softmax 이전에 더해지므로
`-inf`는 확률 0이 된다.

```cpp
out.attention_mask[k + S * (q + S * b)] = k < length ? zero : neg_inf;
```

이 마스크는 `GraphInputs::kq_mask`로 흘러 BERT의 `build_encoder_block`과
Gemma3의 `build_gqa_attention`에 전달된다. M4에서는 두 곳 모두
`/*kq_mask=*/nullptr`이 하드코딩돼 있었다.

`InputRequirements`에 `uses_kq_mask` 비트가 생겼고, 스트리밍에서는 이 비트가
켜진 유닛이 포함된 그룹만 마스크 텐서를 만든다. 어텐션이 없는 FFN 그룹은
마스크를 받지 않는다.

### 4.4 패딩 인지 풀링

풀링은 `[H,S,B]`를 `[H,B]`로 줄인다. 여기서도 패딩을 빼야 한다
([`src/forward/pool.cpp`](../src/forward/pool.cpp)).

- **CLS**: 0번 토큰만 집는다. 패딩은 항상 뒤쪽(right padding)이므로 **영향 없음**.
  마스크가 필요 없다.
- **Mean**: `valid_mask`를 곱해 패딩을 0으로 만들고, `sum_rows`로 더한 뒤,
  `mean_scale`(= 1/실제길이)을 곱한다. 패딩 칸 수로 나누는 사고를 막는다.
- **LAST**: 문장마다 마지막 실제 토큰의 위치가 다르다. `[H, S*B]`로 평탄화한
  뒤 `last_indices`로 `ggml_get_rows` 한 번에 문장별로 다른 행을 집는다.

`padded=false`면 세 경우 모두 기존 무-마스크 구현으로 되돌아간다.

## 5. 병렬 처리는 어디에서 일어나는가

"배치"와 "병렬"은 다른 축이다. NanoEmbed에는 병렬성이 세 층위로 있고,
**M5가 추가한 것은 셋 중 어느 것도 아니다.**

### 층위 1 — 연산 내부 스레드 (ggml CPU 백엔드)

실제 CPU 병렬성은 여기 하나뿐이다.

```cpp
ggml_backend_cpu_set_n_threads(sc.backend, resolve_n_threads(cfg.n_threads));
```

ggml이 행렬곱 하나를 스레드 N개로 쪼갠다. `n_threads=0`이면
`std::thread::hardware_concurrency()`로 결정한다. 벤치마크는 `threads: 4`로 고정했다.

### 층위 2 — 컨텍스트 동시성

컨텍스트 하나는 **한 번에 한 스레드**만 써야 한다. 대신 **컨텍스트 여러 개를
서로 다른 스레드에서 동시에** 돌리는 것은 지원한다. 각 컨텍스트가 자기 백엔드,
자기 그래프 할당자, 자기 `SlotStore`를 갖기 때문이다.

모델 하나를 여러 컨텍스트가 공유하는데, 스트리밍 모드에서 공유되는 가변 상태는
`ResidencyCoordinator` 하나다. 이것은 `mutex`로 보호된다
([`src/streaming_execution.cpp`](../src/streaming_execution.cpp)).
`acquire`/`release`/`retain_common`이 모두 `lock_guard`를 잡는다.

[`tests/integration/context_concurrency_test.cpp`](../tests/integration/context_concurrency_test.cpp)가
이 계약을 검증한다. M5에서 두 컨텍스트가 **동시에 실제 배치**를 돌려도 각자의
순차 기준값과 코사인 0.99999 이상으로 일치하는지 확인하는 검사가 추가됐다.

### 층위 3 — 배치는 병렬이 아니라 폭이다

`B`를 키우는 것은 스레드를 늘리는 게 아니다. **같은 GEMM을 더 넓게** 만들고,
**가중치를 한 번 읽어 여러 문장이 쓰게** 하는 것이다. 이득은 두 가지다.

1. 가중치 I/O 재사용 — 스트리밍에서 lease 횟수가 sub-batch 수에 비례
2. GEMM이 커져 산술 강도(arithmetic intensity)가 올라감

비용도 두 가지다.

1. 패딩 토큰 계산 — 버려질 값에 대한 실제 FLOP
2. 활성값 메모리가 `S*B`로 증가

M5 측정에서는 이 workload에서 **비용이 이득을 넘었다.** 배치 크기를 키울수록
처리량이 단조 감소한다. 자세한 수치는 매트릭스 문서에 있다.

## 6. 중간 상태는 어디에 저장되는가

이 질문의 답은 실행 모드에 따라 완전히 다르다.

### 6.1 eager 모드: 전부 한 그래프 안에

eager는 임베딩부터 풀링까지를 **단일 ggml 그래프**로 만든다. 레이어 사이의
활성값은 그래프 내부의 중간 텐서일 뿐이고, `ggml_gallocr`가 컨텍스트 소유
버퍼 안에 배치한다.

```
ggml_context (메타데이터)  ← 호출마다 새로 만들고 free
ggml_gallocr + 백엔드 버퍼  ← 컨텍스트 수명 동안 재사용
```

메타데이터 아레나만 호출마다 새로 열고 닫는다. 실제 활성값 버퍼는 컨텍스트에
남아 재사용되므로 매번 새로 page fault를 내지 않는다. `gallocr`는 그래프의
노드 수명을 분석해 겹치지 않는 텐서끼리 같은 메모리를 재사용한다. 그래서
레이어 수가 늘어도 활성값 메모리는 레이어 수에 비례하지 않는다.

M3.5에서 확립한 이 구조를 M5는 **바꾸지 않았다.** 달라진 것은 그래프를 만들 때
넘기는 `B`와 `padded`뿐이다.

```cpp
const GraphIO io = impl_->build_graph(gctx, batch.seq_len, batch.batch_size,
                                      batch.padded, cfg.pooling, cfg.normalize);
```

### 6.2 streaming 모드: SlotStore가 그래프 경계를 넘긴다

스트리밍은 전체 그래프를 만들지 않는다. 모델을 **실행 단위(unit)**로 잘라
그룹 단위로 따로따로 실행한다. 그래야 지금 필요한 레이어의 가중치 페이지만
상주시킬 수 있다.

문제는 그래프가 끊기면 ggml 텐서도 같이 사라진다는 것이다. 그룹 A의 출력을
그룹 B가 읽어야 하는데, 그 사이에 `ggml_free(gctx)`가 일어난다.

그래서 **`SlotStore`**가 있다 ([`src/streaming_execution.cpp`](../src/streaming_execution.cpp)).
그래프 경계를 넘는 값을 호스트 메모리(`std::vector<float>`)에 복사해 둔다.

```cpp
struct Entry {
    std::vector<float> data;
    int64_t            ne[4] = {0, 0, 0, 0};
    bool               valid = false;
};
```

동작은 다음과 같다.

```
그룹 실행 → 출력 텐서를 download(slot, tensor)로 호스트 벡터에 복사
          → ggml_free(그룹 컨텍스트)
다음 그룹 → create_input(slot, ctx)로 같은 모양의 입력 텐서 생성
          → set_tensor로 호스트 벡터를 다시 업로드
```

설계상 눈여겨볼 점 셋:

**① 형태를 함께 저장한다.** `ne[4]`를 같이 들고 있어서 다음 그룹이 정확히
같은 모양의 입력 텐서를 만들 수 있다. M5에서 `[H,S,B]`가 되면서 이 부분이
자동으로 B를 따라간다 — `download`가 텐서의 실제 `ne`를 복사하기 때문에
슬롯 코드 자체는 배치를 몰라도 된다.

**② 유효 비트로 stale read를 막는다.** sub-batch가 바뀔 때마다
`invalidate_all()`을 호출한다. capacity는 유지하되 `valid=false`로 만든다.

```cpp
void invalidate_all() noexcept {
    for (Entry & entry : entries_) entry.valid = false;
}
```

이렇게 하면 "이번 sub-batch에서 아무도 쓰지 않은 슬롯을 누가 읽었다"는 상황이
**조용한 이전 값 읽기**가 아니라 예외가 된다. `create_input`이 `valid`를 검사해
`"streaming group reads a slot nothing produced"`를 던진다. 파티션 전략을
바꿔가며 그룹 경계를 재구성하는 코드에서 이런 종류의 버그는 결과가
그럴듯해 보이기 때문에 조용히 넘어가면 잡기 어렵다.

capacity를 유지하는 이유는 sub-batch마다 벡터를 다시 할당하지 않기 위해서다.

**③ F32 연속 텐서만 받는다.** `download`가 타입과 연속성을 검사한다. 슬롯을
통과하는 값의 표현을 하나로 고정해두면 M6의 활성값 압축이 들어올 자리가 명확해진다.

이 복사는 공짜가 아니다. `activation_copy_bytes` 진단 카운터가 얼마나
복사했는지 누적한다. 이것이 스트리밍이 eager보다 느린 이유 중 하나이며,
파티션을 잘게 쪼갤수록(`unit`) 경계가 늘어 복사도 늘어난다. 측정에서
`unit` 파티션이 일관되게 가장 느린 것이 이 비용이다.

### 6.3 sub-batch 사이에는 무엇이 남는가

sub-batch는 서로 완전히 독립이다. 남는 것은 **재사용되는 그릇뿐**이다.

| 항목 | sub-batch 간 |
|---|---|
| `SlotStore` 벡터 capacity | 유지 (내용은 invalidate) |
| `gallocr` 버퍼 | 유지 |
| 백엔드 | 유지 |
| 그래프 메타데이터 컨텍스트 | 매번 새로 만들고 free |
| 활성값 내용 | 무효화 |
| 결과 | 즉시 `output`으로 scatter |

결과를 sub-batch마다 바로 원래 위치에 흩뿌리기 때문에, 전체 배치 결과를
따로 모아두는 버퍼가 없다. `B*H` 크기의 임시 staging 하나만 쓴다.

## 7. 스트리밍 임대(lease)와 실패 처리

스트리밍에서 가중치 페이지는 `madvise`로 상주시켰다가 놓아준다. 이 구간을
`ResidencyCoordinator::Lease`가 RAII로 관리한다.

여기에 M5 코드 리뷰에서 잡힌 결함이 있었다. 원래 순서는 이랬다.

```
① lease 획득
② 그래프 메타데이터 컨텍스트 생성   ← 여기서 실패하면?
```

②가 실패하면 스코프를 빠져나가며 lease 소멸자가 `compute_complete=false`
상태로 release를 시도한다. 그러면 coordinator가 **poison** 상태가 되고
(`premature_release_attempts` 증가), 이후 모든 `acquire`가 실패한다.
문장 하나의 할당 실패가 모델 핸들 전체를 망가뜨린다.

수정은 순서를 뒤집는 것이었다.

```
① 실패할 수 있는 호스트 할당을 먼저 (그래프 컨텍스트, 토큰 range 계산)
② 그 다음에 lease 획득
③ lease 획득 이후의 동기 실패는 mark_compute_complete()로 표시 후 정리
```

③이 안전한 이유는 백엔드 실행이 **동기**이기 때문이다. `ggml_backend_graph_compute`가
반환했거나 예외를 던졌다면 매핑된 페이지를 더 읽는 주체가 없다. 그래서
"계산이 끝났다"고 표시하고 놓아주는 것이 정확하다.

```cpp
auto lease = impl_->residency->acquire("token", rows, true);
try {
    /* ... 그래프 구축, 실행, 슬롯 download ... */
    lease.mark_compute_complete();
    lease.release();
} catch (...) {
    lease.mark_compute_complete();
    throw;
}
```

이 경계를 결정적으로 시험하기 위해 **의도적 실패 주입**이 들어있다.

```cpp
void InternalStreamingContext::diagnostic_fail_graph_context_after(uint64_t n);
```

`n`번째 성공 이후 다음 그래프 컨텍스트 생성이 한 번 실패한다. 공개 C ABI
바깥의 내부 함수다. 테스트는 임베딩 컨텍스트 실패와 첫 그룹 컨텍스트 실패를
각각 주입해 다음을 확인한다.

- 배치 출력 전체가 NaN
- active lease가 0
- coordinator가 poison 상태가 **아님**
- 같은 컨텍스트를 실패 직후 재사용해도 eager 배치와 결과가 일치

## 8. 오류 계약

`nanoembed_embed_batch`의 계약은 헤더에 명시되어 있다.

```
인자가 잘못된 경우      → out을 건드리지 않음
실행이 실패한 경우      → out 전체를 NaN으로 오염
```

이 구분에는 이유가 있다. 인자 검증은 실행 **전에** 끝나므로 출력 버퍼를
건드릴 이유가 없다. 반면 실행 중 실패는 이미 일부 sub-batch가 결과를 쓴
뒤일 수 있다. 이때 부분 결과를 남기면 호출자가 "일부는 유효한 벡터"라고
착각할 수 있는데, 어느 것이 유효한지 알 방법이 없다. 그래서 전부 NaN으로
만든다. NaN은 코사인 유사도 계산에 넣으면 즉시 전파되므로 조용히 넘어가지 않는다.

예외는 다음과 같이 매핑된다 ([`src/api/c_api.cpp`](../src/api/c_api.cpp)).

| 예외 | 반환 코드 |
|---|---|
| `TokenizerError` | `NANOEMBED_ERR_TOKENIZE` |
| `AllocationError` | `NANOEMBED_ERR_OOM` |
| `std::bad_alloc`, `std::length_error` | `NANOEMBED_ERR_OOM` |
| 그 외 `std::exception` | `NANOEMBED_ERR_INTERNAL` |

`AllocationError`는 M5에서 새로 만든 예외 타입이다. 이전에는 할당 실패가
`std::runtime_error`로 뭉뚱그려져 `ERR_INTERNAL`이 됐는데, OOM과 내부 버그는
호출자가 다르게 대응해야 한다.

`n_texts == 0`은 오류가 아니라 성공이다. 아무것도 쓰지 않고 `NANOEMBED_OK`를 반환한다.

## 9. 정확도는 어떻게 검증했는가

[`tests/integration/batch_test.cpp`](../tests/integration/batch_test.cpp)가
네 모델(BERT F16, Harrier F32/Q8/Q4)에 대해 확인한다.

1. **계획 단계 단위 검사** — sub-batch 개수, 정렬 후 원래 인덱스, 패딩 칸 수,
   `valid_mask`, `last_indices`를 손으로 계산한 값과 대조
2. **배치 대 순차 대조** — 같은 입력을 순차 경로와 배치 경로로 돌려 비교
3. **같은 길이 배치** — 마스크 없는 경로가 순차와 **정확히 0 차이**인지
4. **경계 크기** — `max_batch` 주변 값들에서 분할이 결과를 바꾸지 않는지
5. **오류 계약** — null 입력에서 sentinel이 보존되는지, `n_texts=0`이 OK인지

3번이 설계 의도를 직접 검증한다. 오차는 오직 패딩된 그래프에서만 발생한다.

패딩된 경로의 오차는 원래 계획한 gate(코사인 ≥0.999999, 최대 절대오차 ≤1e-5)를
통과하지 못했다. BERT F16에서 최대 절대오차 2.345e-4가 관측됐다. 원인은
두 가지로 본다.

- 패딩된 그래프의 softmax 행이 더 넓다 (`-inf`가 들어가도 행 길이가 다름)
- ARM 벡터 리덕션의 그룹핑이 `S*B` 평탄화로 달라진다

회귀 테스트는 관측값에 근거한 코사인 ≥0.999998, 최대 절대오차 ≤2.5e-4를
**명시적으로** 사용한다. 이것은 원 계약의 승인된 대체가 아니라 현재 상태의
기록이다. 이 완화를 승인할지, 아니면 패딩 커널 parity를 고칠지는
[`docs/m5-review-followups.md`](m5-review-followups.md)에 미결로 남아 있다.

## 10. 의도적으로 넣지 않은 것

수치를 좋게 만들 수 있었지만 넣지 않은 것들이다. 이유를 함께 남긴다.

| 넣지 않은 것 | 이유 |
|---|---|
| 토큰 예산 기반 적응형 분할 | `max_batch`가 유일한 분할 기준이라는 계약을 우회함 |
| 배치가 느릴 때 eager로 자동 fallback | 숨은 모드 전환은 측정 결과를 재현 불가능하게 만듦 |
| 환경변수 기반 튜닝 스위치 | 위와 같음 |
| 실패 시 자동 재시도 | 실패를 감춰 계약을 모호하게 만듦 |
| token packing | M5 범위 밖. 별도 실험으로 측정해야 함 |

이 중 token-budget 또는 길이 편차 기반 분할은 **다음 최적화 후보**로 기록돼
있다. 측정 결과가 패딩 비용이 지배적임을 보여주기 때문이다.

## 11. 요약

- `batch.{h,cpp}`가 토큰화·정렬·패딩·마스크를 담당하고, eager와 streaming이 공유한다
- 길이 stable sort로 패딩을 줄이고, `original_index`로 입력 순서를 복원한다
- 같은 길이 sub-batch는 마스크 없는 M4 그래프를 그대로 쓴다 (오차 0)
- `build_linear`가 `[S,B]`를 평탄화해 공유 가중치를 한 GEMM으로 처리한다
- 병렬성은 ggml 스레드(연산 내부)와 컨텍스트 동시성 두 가지이며, 배치는 병렬이 아니라 폭이다
- eager는 중간 상태를 한 그래프 안 `gallocr` 버퍼에 두고, streaming은 `SlotStore`가
  호스트 메모리로 그래프 경계를 넘긴다
- lease는 실패할 수 있는 할당이 모두 끝난 뒤에 획득한다
- 실행 실패는 출력 전체를 NaN으로 만든다
