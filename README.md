# multiuser-chat

This workspace builds a small supervised fine-tuning experiment for implicit multi-user routing.

Pipeline:

1. Filter short, simple, multi-turn human-to-AI conversations from `allenai/WildChat`.
2. Assign each source conversation a phone number.
3. Interleave user/assistant turn pairs while preserving each conversation's internal order.
4. Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` with LoRA.
5. Compare baseline and fine-tuned models on held-out interleaved conversations.

The assistant target never includes chain-of-thought and never includes the phone prefix.

## Results

See [`COMPARISON.md`](COMPARISON.md) for the base-model versus fine-tuned behavior report.

W&B run: https://wandb.ai/anothervibecoder-i-unemplyed/multiuser-phone-sft/runs/2l8h27ni

Note: `wandb sync` produced the run URL above, but reported a 403 while uploading `wandb-metadata.json`; local W&B logs remain under `wandb/offline-run-20260618_061706-2l8h27ni`.

Commands:

```bash
python prepare_data.py
python eval_model.py --out runs/baseline_eval.json
python train_lora.py --wandb --wandb-project multiuser-chat --wandb-run-name qwen-phone-lora
python eval_model.py --adapter runs/qwen-phone-lora --out runs/finetuned_eval.json
python write_report.py --baseline runs/baseline_eval.json --finetuned runs/finetuned_eval.json --trainer-metrics runs/qwen-phone-lora/final_metrics.json --out runs/behavior_report.md
```

Phone prefixes are generated as realistic-looking NANP numbers, avoiding the fictional `555`
exchange.

If W&B is not logged in on the machine, `train_lora.py --wandb` automatically uses W&B
offline mode and writes local run logs under `wandb/`.
