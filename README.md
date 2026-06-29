# Experimental Results

|  | GSM8K (Exact Match) | SST2 (Accuracy) |
| --- | --- | --- |
| vanilla LoRA | 0.2397 | 0.9615 |
| Ours | 0.4256 | 0.9677 |

The results of training the masked LoRA are as follows. The table reports the average performance over 16 seeds on the GSM8K and SST2 datasets, using Llama-3.2-3B as the backbone model under the LoRA rank=16, alpha=32 setting.

From the experimental results, we can observe a significant performance difference between vanilla LoRA and Ours under the same environmen
