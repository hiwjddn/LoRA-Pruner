# 실험 결과

|  | GSM8K (Exact Match) | SST2 (Accuracy) |
| --- | --- | --- |
| vanilla LoRA | 0.2397 | 0.9615 |
| Ours | 0.4256 | 0.9677 |

Masked LoRA를 학습한 결과는 다음과 같다. 표는 Llama-3.2-3B를 백본 모델로 사용하고, LoRA rank=16, alpha=32 설정에서 GSM8K와 SST2 데이터셋에 대해 16개 seed의 평균 성능을 보고한다.

실험 결과를 통해, 동일한 환경에서 vanilla LoRA와 Ours 사이에 유의미한 성능 차이가 있음을 확인할 수 있다.
