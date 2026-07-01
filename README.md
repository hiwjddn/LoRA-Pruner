## 코드 재현 
### Vanilla LoRA 학습
```bash
python src/train_vanilla_lora.py \
  --model meta-llama/Llama-3.2-3B \
  --task sentiment \
  --output_dir results/lora_llama_sent/r16a32/1
```

### Mast 생성
```bash
python src/make_lora_mask.py \
  --metric_jsonl results/lora_llama_sent/r16a32/1/compare_lora.jsonl \
  --output_path results/mask/lora_llama_sent/r16a32/1/mask.pt
```

### Masked LoRA 학습 
```bash
python src/train_masked_lora.py \
  --model meta-llama/Llama-3.2-3B \
  --task sentiment \
  --mask_path results/mask/lora_llama_sent/r16a32/1/mask.pt \
  --output_dir results/masked_lora_llama_sent/r16a32/1
```

### Masked LoRA 평가 
```bash
python src/eval_lora.py \
  --base_model meta-llama/Llama-3.2-3B \
  --lora_path results/masked_lora_llama_sent/r16a32/1 \
  --task sentiment
```
## 실험 결과
|  | GSM8K (Exact Match) | SST2 (Accuracy) |
| --- | --- | --- |
| vanilla LoRA | 0.2397 | 0.9615 |
| Ours | 0.4256 | 0.9677 |

Masked LoRA를 학습한 결과는 다음과 같다. 표는 Llama-3.2-3B를 백본 모델로 사용하고, LoRA rank=16, alpha=32 설정에서 GSM8K와 SST2 데이터셋에 대해 16개 seed의 평균 성능을 보고한다.
실험 결과를 통해, 동일한 환경에서 vanilla LoRA와 Ours 사이에 유의미한 성능 차이가 있음을 확인할 수 있다.
