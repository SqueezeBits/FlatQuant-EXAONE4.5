# vLLM AWQ vs. FlatQuant W4A16 Inference Path

이 문서는 EXAONE-4.5-33B에서 original vLLM AWQ와 현재 FlatQuant W4A16
vLLM plugin의 inference 경로 차이, 성능 및 메모리 비용, 최적화 방향을
정리한다.

## 1. 결론

두 경로는 vLLM scheduler, KV cache, attention backend, fused projection,
W4A16 Marlin GEMM을 공유한다. FlatQuant W4A16의 hot path에 추가되는 핵심
수학 연산은 각 projection 입력의 learned transform이다.

```text
AWQ:
    BF16 activation
      -> Marlin INT4 x BF16 Linear
      -> BF16 output

FlatQuant W4A16:
    BF16 activation
      -> learned online transform
      -> Marlin INT4 x BF16 Linear
      -> BF16 output
```

W4A16에서는 activation fake quantization과 KV-cache quantization을 하지
않지만, weight가 transformed coordinate system에서 INT4로 양자화됐으므로
현재 checkpoint의 정확도를 유지하려면 online transform은 필요하다.

## 2. 공통으로 사용하는 vLLM 경로

| 구성 요소 | AWQ | FlatQuant W4A16 |
|---|---|---|
| vLLM V1 scheduler | 동일 | 동일 |
| Continuous batching | 동일 | 동일 |
| PagedAttention / KV cache | 동일 | 동일 |
| FlashAttention | 동일 | 동일 |
| QKV projection fusion | 사용 | 사용 |
| gate/up projection fusion | 사용 | 사용 |
| Activation dtype | BF16 | BF16 |
| Weight dtype | INT4 | INT4 |
| Weight group size | 128 | 128 |
| Weight GEMM | vLLM Marlin WNA16 | 동일한 Marlin WNA16 |
| Sampling / tokenizer | 동일 | 동일 |

FlatQuant plugin은 vLLM의 compressed-tensors WNA16 quant method를 내부에서
그대로 호출한다. 따라서 처리량 차이는 다른 attention engine이나 다른
INT4 GEMM을 사용해서 생기는 것이 아니다.

## 3. 실제로 달라지는 projection 경로

한 transformer layer에서 FlatQuant가 추가하는 transform은 다음과 같다.

```text
Attention
  hidden_states
    -> qkv Kronecker transform
    -> fused qkv_proj (Marlin)

  attention output
    -> o_proj head transform
    -> o_proj (Marlin)

MLP
  hidden_states
    -> up/gate Kronecker transform
    -> fused gate_up_proj (Marlin)

  intermediate activation
    -> down Kronecker transform
    -> down_proj (Marlin)
```

현재 plugin에서 transform을 적용하는 projection suffix는 다음 네 개다.

- `qkv_proj`
- `gate_up_proj`
- `down_proj`
- `o_proj`

관련 구현:

- `vllm_plugin/flatquant_vllm_plugin/config.py`: compressed-tensors/Marlin
  quant method wrapper
- `vllm_plugin/flatquant_vllm_plugin/transform.py`: projection별 transform dispatch
- `vllm_plugin/flatquant_vllm_plugin/triton_transform_v2.py`: Triton transform kernel
- `tools/export_flatquant_vllm.py`: FlatQuant checkpoint를 vLLM WNA16 storage로 변환

## 4. W4A16에서도 transform이 필요한 이유

FlatQuant PTQ 코드의 activation quantizer는 `bits == 16`일 때 identity다.

```python
if self.bits == 16 or not self.enable:
    return x
```

하지만 EXAONE adapter는 다음 조건으로 transform을 만든다.

```python
if self.args.w_bits < 16 or self.args.a_bits < 16:
    # create learned transforms
```

즉 transform은 activation quantization만을 위한 것이 아니다. PTQ의
`reparameterize()`는 inverse transform을 weight에 적용하며, 그 transformed
weight를 INT4로 양자화한다.

개념적인 연산은 다음과 같다.

```text
W' = T^-1 W
y_flatquant = T(x) Q(W')
              = T(x) Q(T^-1 W)
```

양자화가 없다면 `T`와 `T^-1`가 상쇄되지만, INT4 quantizer `Q`가 사이에
있으므로 일반적으로 다음 두 연산은 같지 않다.

```text
T(x) Q(T^-1 W) != x Q(W)
```

따라서 현재 checkpoint에서 transform만 제거하면 activation과 weight의
coordinate system이 맞지 않는다.

실제 W4A16 checkpoint 설정도 `a_bits`, `q_bits`, `k_bits`, `v_bits`는 모두
16이지만 다음 online transform을 명시한다.

```json
{
  "w_bits": 4,
  "a_bits": 16,
  "q_bits": 16,
  "k_bits": 16,
  "v_bits": 16,
  "online_trans": [
    "qk",
    "o_proj",
    "down_proj",
    "qkv_proj",
    "up_gate_proj"
  ]
}
```

## 5. 현재 구현에서 추가되는 비용

현재 Kronecker transform은 right와 left factor를 순서대로 적용한다.

```text
x
  -> right-transform Triton kernel
  -> BF16 temporary buffer
  -> left-transform Triton kernel
  -> BF16 transformed activation
  -> Marlin
```

transform 수학 연산 외에 다음 구현 비용이 추가된다.

- projection별 추가 Triton kernel launch
- right/left 단계 사이 BF16 temporary buffer
- transformed activation 출력 buffer
- temporary buffer의 global-memory write/read
- activation 추가 read/write에 따른 memory bandwidth 사용
- eager mode의 Python/CUDA launch overhead

`o_proj`는 learned right factor가 없으므로 현재 최적화된 경로에서는
identity matrix를 만들거나 right-transform kernel을 실행하지 않고
left-only kernel을 사용한다.

## 6. 메모리 차이

### 정적 메모리

| 항목 | AWQ | FlatQuant W4A16 |
|---|---:|---:|
| INT4 weights | 거의 동일 | 거의 동일 |
| Group scales | 거의 동일 | 거의 동일 |
| KV cache | 동일 | 동일 |
| Learned transform factors | 없음 | 추가 |

Transform은 full dense hidden-size matrix가 아니라 Kronecker factor로
저장되므로 전체 INT4 model weight에 비해 정적 용량은 작다.

### 동적 메모리

Transform buffer 하나의 대략적인 크기는 다음과 같다.

```text
tokens x input dimension x 2 bytes (BF16)
```

Decode batch 1에서는 작지만, 긴 prefill과 큰 MLP intermediate dimension에서는
수십~수백 MB가 될 수 있다. 모든 layer buffer가 동시에 유지되는 것은
아니지만, 현재 two-stage transform은 projection 실행 중 입력 크기의
temporary buffer를 추가로 요구한다.

## 7. 이전 최적화 단계 측정 결과 (historical)

측정 조건:

- GPU: NVIDIA A100 80GB
- batch size: 1
- prefill: 64 tokens
- decode: 16 tokens
- dtype: BF16
- eager mode
- warmup 1회, measurement 3회

| 경로 | Output throughput |
|---|---:|
| original vLLM AWQ | 26.26 token/s |
| FlatQuant Triton v1 | 15.03 token/s |
| FlatQuant Triton v2 | 16.03 token/s |
| FlatQuant v2 + `o_proj` left-only | 17.11 token/s |
| FlatQuant fused Kronecker (단계 4) | 18.36 token/s |

`o_proj`의 identity allocation과 불필요한 right kernel을 제거한 결과,
16.03에서 17.11 token/s로 약 6.7% 개선됐다.

단계 4의 fused Kronecker kernel end-to-end 재측정 (같은 A100 machine에서
two-stage와 fused를 동일 조건으로 재실행, 동일 checkpoint):

| 조건 | two-stage | fused | 개선 |
|---|---:|---:|---:|
| prefill 64 / decode 16 / bs 1 | 16.72 token/s | 18.36 token/s | +9.8% |
| prefill 512 / decode 16 / bs 4 | 34.64 token/s | 41.20 token/s | +18.9% |

decode 중심 config에서 약 +9.8%, prefill/batch가 커지면 temp buffer 제거와
bandwidth 절감 효과가 커져 +18.9%로 격차가 확대된다. two-stage baseline은
`apply_transform`이 `flatquant_kron_transform_two_stage`를 호출하도록
임시로 바꿔 동일 machine에서 재현했다.

위 수치는 모두 `--enforce_eager` 기준이다.

### CUDA Graph 활성화 (단계 3)

transform을 torch custom op으로 등록한 뒤 `--enforce_eager`를 제거하면
vLLM이 model 전체를 하나의 graph(compile range 1-8192)로 compile하고 FULL
decode + PIECEWISE CUDA graph를 capture한다. graph break나 shape
specialization 오류는 없었다 (capture: FULL decode 35개 + PIECEWISE 51개,
1.07 GiB, 38초). 동일 decode config(prefill 64 / decode 16 / bs 1):

| 경로 | Output throughput |
|---|---:|
| FlatQuant fused, eager | 18.36 token/s |
| FlatQuant fused, CUDA Graph | 59.21 token/s |

CUDA Graph로 eager 대비 약 3.2x 빨라졌다. 이는 transform FLOPs가 아니라
eager mode의 per-op Python/CUDA launch overhead가 decode의 지배적 비용이었음을
보여준다. 생성 텍스트도 정상적으로 유지됐다 (correctness gate 통과).

### AWQ vs FlatQuant 공정 비교 (둘 다 CUDA Graph)

AWQ도 동일 조건에서 non-eager로 재측정해 두 경로를 CUDA Graph 기준으로
비교했다 (prefill 64 / decode 16 / bs 1, warmup 2 / measurement 3):

| 경로 (non-eager) | Output throughput | AWQ 대비 |
|---|---:|---:|
| original vLLM AWQ | 65.04 token/s | 100% |
| FlatQuant fused W4A16 | 59.21 token/s | 91.0% |

eager 기준 격차(17.11 대 26.26, AWQ 대비 65%)가 CUDA Graph에서는 AWQ 대비
91%까지 좁혀졌다. 남은 약 9%는 각 projection의 learned transform이 추가하는
FLOPs와 별도 kernel로, 현재 checkpoint 정확도를 유지하는 한 알고리즘적으로
남는 비용이다 (단계 5의 transform+Marlin fusion으로만 추가 축소 가능).

이 결과는 작은 batch의 decode 중심 smoke benchmark다. 긴 prefill 및 여러
동시 request에서는 transform FLOPs와 memory traffic의 상대적 비중이 달라질
수 있으므로 별도의 benchmark가 필요하다.

## 8. 제거 가능한 비용과 제거 불가능한 비용

| 비용 | 제거 가능성 | 해결 방법 |
|---|---|---|
| Python dispatch overhead | 완료 (대부분 제거) | torch custom op 등록 + CUDA Graph |
| CUDA Graph 미적용 비용 | 완료 (제거) | dynamic-shape-safe custom op 및 graph capture |
| right/left 두 번의 launch | 완료 (한 번으로 축소) | fused Kronecker Triton kernel |
| right/left 사이 temporary buffer | 완료 (제거) | register-resident intermediate |
| temporary global-memory traffic | 완료 (제거) | 같은 fused kernel에서 두 factor 처리 |
| transformed activation buffer | 조건부 제거 가능 | transform + Marlin fusion |
| transform 자체의 FLOPs | 제거 불가 | 현재 INT4 checkpoint 정확도 유지에 필요 |

## 9. 최적화 계획

### 단계 1: 최소 변경 baseline 고정

AWQ용 vLLM runner를 그대로 사용하고 model path만 FlatQuant checkpoint로
변경한다.

```bash
/workspace/.venvs/flatquant-vllm/bin/python \
  benchmarks/exaone45/vllm_awq.py generate \
  --model_path \
    outputs/EXAONE-4.5-33B/w4a16-vllm/exaone45-33b-w4a16-vllm \
  --enforce_eager
```

이 baseline에서는 scheduler, attention, Marlin, sampling 경로가 AWQ와
동일하고 FlatQuant transform wrapper만 추가된다.

### 단계 2: projection별 불필요한 연산 제거

- [x] `o_proj` identity matrix allocation 제거
- [x] `o_proj` right-transform kernel 제거
- [ ] transform parameter loader warning 정리
- [ ] contiguous copy가 실제로 필요한 shape만 선별

### 단계 3: CUDA Graph 활성화 (완료)

첫 시도에서는 Triton launch grid가 dynamic token dimension을 8192로
specialize해 `torch.compile` shape constraint 오류가 발생했다.

해결 및 검증 결과:

- [x] transform을 torch custom op(`flatquant::kron_transform`,
  `flatquant::left_transform`)으로 등록해 Dynamo에 opaque operation으로 노출
  (`triton_transform_v2.py`). `apply_transform`은 `kron_transform` /
  `left_transform` wrapper를 통해 이 op을 호출한다.
- [x] `register_fake`로 shape-preserving fake/meta implementation 제공
  (`torch.empty_like(x)`).
- [x] token dimension은 launch grid의 첫 축에만 쓰이고 `tl.constexpr`로는
  layer-static factor 크기(`L_PAD`/`R_PAD`/`BLOCK_N`)만 전달하므로 dynamic
  shape-safe다. `torch.compile(fullgraph=True, dynamic=True)`가 token 수
  1/5/37/128에서 graph break 없이 통과함을 단위 수준에서 확인.
- [x] `--enforce_eager` 없이 end-to-end 실행 시 FULL decode graph(size 1
  포함) 35개 + PIECEWISE graph 51개가 정상 capture되고 (§7), decode
  throughput이 18.36 -> 59.21 token/s로 약 3.2x 개선됐다.
- [x] prefill/decode dynamic token range: compile range (1, 8192) 단일
  graph로 compile되고 graph break/recompile 없음.

CUDA graph 사용 시에도 생성 텍스트는 정상적으로 유지됐다.

### 단계 4: right + left transform fusion (완료)

목표 경로:

```text
x
  -> fused Kronecker Triton kernel
  -> transformed activation
  -> Marlin
```

`triton_transform_v2.py`의 `_fused_kron_kernel`이 이 경로를 구현한다. 각
program은 하나의 batch row와 right 차원의 `BLOCK_N` slice를 담당하고,
`x @ right` 결과의 전체 `L` row를 register에 유지한 뒤 같은 launch에서
left factor를 적용한다. 이를 통해 kernel launch 하나와 right/left 사이의
temporary global buffer 하나를 제거한다. 기존 two-stage 경로는
`flatquant_kron_transform_two_stage`로 A/B 비교용으로 남겨 뒀고,
`apply_transform`은 fused 경로를 호출한다.

검증 결과 (A100 80GB, BF16, `tests/test_triton_transform.py`의
`FusedKronTransformTest`):

- fused 출력은 fp32 reference 대비 relative max error ≤ 5e-3이며,
  fused가 register에 fp32 중간값을 유지하므로 two-stage(BF16 temp buffer
  왕복)보다 오차가 크지 않다.
- kernel latency (fused vs two-stage):

  | shape | tokens=1 | tokens=64 | tokens=512 | tokens=2048 |
  |---|---:|---:|---:|---:|
  | qkv/gate_up (64x80) | 1.47x | 1.49x | 2.22x | 3.41x |
  | down_proj (128x214) | 1.32x | 3.33x | 4.33x | 5.45x |

decode(tokens=1)에서는 launch 수 감소로 1.3~1.5x, 긴 prefill에서는 temp
buffer 제거와 bandwidth 절감으로 down_proj 기준 최대 5.45x 빨라졌다.

end-to-end vLLM token/s 재측정은 §7에 정리했다 (decode +9.8%, prefill +18.9%).

남은 검증 항목:

- peak GPU memory 및 achieved memory bandwidth 프로파일

### 단계 5: transform + Marlin fusion 타당성 평가

이론적인 목표는 다음과 같다.

```text
x
  -> fused transform + INT4 GEMM
  -> output
```

이 경우 transformed activation buffer까지 제거할 수 있다. 하지만 transform은
Marlin GEMM보다 앞에 있으며 K dimension reduction을 포함한다. 단순히 Marlin
tile마다 transform을 재계산하면 output tile 수만큼 계산이 중복되어 오히려
느려질 수 있다.

따라서 다음 조건을 확인한 뒤 진행한다.

- transformed input tile을 여러 output tile에서 재사용할 수 있는가
- Marlin의 input staging/K-loop에 transform을 넣을 수 있는가
- register/shared-memory 사용 증가가 occupancy를 크게 낮추지 않는가
- separate fused-transform + Marlin보다 실제로 빠른가

이 단계는 기존 Marlin을 그대로 사용하는 plugin 수준 최적화가 아니라
custom Marlin kernel 수정에 해당한다.

판정은 **NO-GO**다. vLLM 0.24.0 Marlin은 N 방향 64--256열 slice를 여러
CTA가 소유하며 각 CTA가 같은 activation의 K tile을 자체 shared memory로
stage한다. 현재 구조 안에 transform을 넣으면 projection당 한 번이 아니라
N slice마다 재계산된다. A100에는 CTA 사이 transformed activation을 공유할
threadblock-cluster shared memory도 없다. 상세 근거와 재검토 조건은
`experiments/marlin_fusion/README.md`에 기록했다.

## 10. 정확도 검증 게이트

속도 최적화 전후에 다음 결과가 유지돼야 한다.

1. Triton transform과 PyTorch reference의 layer-wise 출력 비교
2. original Transformers FlatQuant runtime과 vLLM logits 비교
3. 짧은 greedy generation token 일치 확인
4. WikiText 또는 동일 calibration/evaluation set의 perplexity 비교
5. downstream lm-eval 결과 비교

현재 converted packed INT4 weight는 original FlatQuant checkpoint와 bit-exact
unpack 비교를 통과했다. 다만 전체 model logits/perplexity를 original
Transformers FlatQuant runtime과 비교하는 검증은 별도의 correctness gate로
남아 있다.

## 11. 현재 도달한 상태

FlatQuant의 transform 수학 연산 때문에 AWQ와 완전히 동일한 비용이 되기는
어렵다. 그러나 초기 eager 격차(26.26 대 17.11 token/s) 전체가 알고리즘적으로
필연적인 것은 아니었다.

단계 3(CUDA Graph)과 단계 4(fused Kronecker)를 적용한 뒤, 동일 조건에서
두 경로를 CUDA Graph로 비교하면 FlatQuant W4A16이 AWQ의 91%(59.21 대 65.04
token/s)에 도달한다. 즉 eager 기준 AWQ 대비 65%였던 처리량이 91%로
올라갔고, 남은 약 9%가 transform FLOPs에서 오는 잔여 비용이다.

달성한 항목:

- decode: torch custom op + CUDA Graph로 launch overhead 제거 (eager 대비 3.2x)
- prefill: fused Kronecker로 temporary memory와 bandwidth 비용 감소
- 공통 vLLM/Marlin 경로 유지
- FlatQuant checkpoint 정확도 유지 (생성 텍스트 정상, 단위 오차 게이트 통과)

남은 항목:

- transform을 제외하고 새로 calibration/quantization한 checkpoint의 품질 평가
- §10의 전체 logits/perplexity correctness gate

최종 성능 판단은 단계별 동일 조건 benchmark와 correctness 검증을 함께
통과한 수치로 한다.

## 12. 2026-07-14 shape-aware A100 최적화 결과

위 §7의 eager/초기 CUDA Graph 수치는 과거 단계 기록이다. 이번 작업에서는
vLLM request metrics를 이용해 TTFT와 per-request TPOT를 분리하고, warmup 2회와
measurement 5회로 다시 측정했다. 환경은 A100 80GB, BF16, vLLM 0.24.0,
CUDA Graph 활성화다.

### 최종 latency

| workload | model | TTFT median | TPOT median | output tok/s |
|---|---|---:|---:|---:|
| bs=1, prefill=64, decode=128 | AWQ | 33.679 ms | 14.266 ms | 69.352 |
| bs=1, prefill=64, decode=128 | FlatQuant baseline | 35.446 ms | 15.668 ms | 63.202 |
| bs=1, prefill=64, decode=128 | FlatQuant tuned | 35.729 ms | 15.267 ms | 64.829 |
| bs=4, prefill=512, decode=16 | AWQ | 710.351 ms | 16.035 ms | 67.288 |
| bs=4, prefill=512, decode=16 | FlatQuant baseline | 778.175 ms | 17.247 ms | 61.713 |
| bs=4, prefill=512, decode=16 | FlatQuant tuned | 774.733 ms | 16.910 ms | 62.246 |

Shape-aware Triton config로 primary TPOT는 2.56% 감소하고 output throughput은
2.57% 증가했다. secondary TPOT는 1.96% 감소하고 throughput은 0.86% 증가했다.
TTFT 변화는 primary에서 +0.283 ms로 측정 노이즈 수준이며 secondary에서는
3.442 ms 감소했다.

선택한 SM80 config는 `(left,right)=(64,80)` decode에서 `BLOCK_N=16`,
`(128,214)` decode에서 `BLOCK_N=16`, 이후 token bucket에 따라 32/64로
확대한다. 알 수 없는 GPU에서는 기존 `BLOCK_N=64, num_warps=4`로 fallback한다.
108개 shape/config GPU sweep은 모두 fp32 reference relative max error `5e-3`
이내였다.

### 프로파일 결론

- CUDA Graph는 FULL decode 35개와 PIECEWISE 51개가 유지됐고 graph break는 없다.
- steady-state CUDA kernel 합산 시간에서 transform은 약 9.4%, Marlin은 약
  74%였다.
- transformed activation 앞의 별도 contiguous/copy hot path는 발견되지 않아
  layout 변경은 하지 않았다.
- tuned FlatQuant와 AWQ의 primary TPOT 차이는 1.001 ms(약 7.0%)다. 이는 현재
  checkpoint 좌표계를 유지하는 transform의 잔여 비용이다.

### 정확성 및 export 안전성

- 기존 `BLOCK_N=64`와 tuned kernel의 32-token greedy 출력은 byte-exact하게
  같았다 (SHA-256
  `e22521031e06d002ff0ea0f1f134d917e3a061794d128dfa9e96eeac0124b079`).
- 관련 benchmark/transform/export 테스트 24개가 통과했다.
- 전체 repository pytest는 테스트 실패 전에 collection 단계에서 현재 vLLM
  환경에 없는 `fast_hadamard_transform`과 `scipy` 때문에 중단됐다.
- 이미 transformed coordinate system으로 pack된 checkpoint에서 transform
  tensor만 제외하는 것은 금지했다. `--exclude-transform`은 projection/layer
  선택을 검증하지만, 실제 제외 요청은 해당 layer를 transform 없이 다시
  calibration하고 W4로 재양자화하기 전까지 fail-closed로 거부한다.

재현 가능한 raw JSON은 로컬 `outputs/benchmark_results/{baseline,tuned}`에,
kernel sweep 결과는 `outputs/benchmark_results/kernels/a100.json`에 저장했다.
