# multiuser-chat

Small SFT experiment for teaching a chat model to handle multiple simultaneous users in one transcript.

The setup is intentionally simple: each user message starts with a phone-number prefix, but the assistant reply does not. The model has to infer that the latest phone number identifies the active conversation, answer only that user, and keep prior context separated by phone number.

This matters because real agents are not always one-user chatbots. Systems like OpenClaw, email agents, support agents, sales agents, or scheduling agents may talk to many people through the same interface at the same time. A human operator can keep separate conversations straight: one email thread is about a refund, another is about a meeting, and another is from a client with different preferences and history. An agent needs that same ability. If it mixes context between people, it might send the wrong information, reuse the wrong tone, leak private details, or answer a follow-up using another person's conversation history.

This repo tests and trains that skill directly: handling interleaved conversations while keeping each user's state separate.

## Example

Single-user conversations are first collected independently:

```text
+1-615-443-1773: please customize this bio: I worked at XYZ company for 6 years as a software developer
assistant: I have gained extensive experience as a software developer during my tenure of over 6 years at XYZ company.

+1-615-443-1773: make it 100 words
assistant: During my tenure of over 6 years at XYZ company, I have flourished as a software developer...
```

Then several conversations are interleaved:

```text
+1-615-443-1773: please customize this bio: I worked at XYZ company for 6 years as a software developer
assistant: I have gained extensive experience as a software developer during my tenure of over 6 years at XYZ company.

+1-919-809-4491: improve this apology email
assistant: Please accept my sincere apologies for the confusion...

+1-615-443-1773: make it 100 words
assistant: During my tenure of over 6 years at XYZ company, I have flourished as a software developer...

+1-919-809-4491: rephrase the first two sentences
assistant: I would like to extend my sincere apologies...
```

The assistant is never trained to print the phone prefix. The phone number is only a routing/context key.

## Data

The data source is `allenai/WildChat`, filtered down to short, simple, multi-turn human-to-AI conversations.

The filter keeps conversations that are:

- English, non-toxic, and not redacted
- 4 to 8 alternating user/assistant messages
- short enough for a small local run
- mostly plain English
- likely to contain a follow-up dependency, such as "make it shorter", "change it", "same", "what about", or another short continuation

The filter drops examples that are likely to distract from the routing task:

- code blocks
- pasted files
- very long essays/scripts
- translation/document-summarization tasks
- prompt-injection style text
- math/proof/step-by-step reasoning prompts
- visible chain-of-thought requests

Phone prefixes are generated as realistic-looking NANP numbers, while avoiding the fictional `555` exchange.

## Model

The run uses `Qwen/Qwen2.5-0.5B-Instruct`.

It is already a chat model and supports a chat template. The fine-tune is LoRA SFT, not full-parameter training:

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Trainable adapter params: about 8.8M
- Local GPU used: RTX 3060 12GB
- Assistant targets: final answer only, no chain-of-thought

## Results

See [`COMPARISON.md`](COMPARISON.md) for the full base-model versus fine-tuned behavior report.

Summary from the held-out interleaved split:

| Metric | Base model | Fine-tuned |
| --- | ---: | ---: |
| Eval loss | 1.4751 | 0.4018 |
| Perplexity | 4.3716 | 1.4945 |
| Mean word F1 | 0.2971 | 0.5928 |
| Phone-prefix violation rate | 0.0000 | 0.0000 |

The fine-tuned model more often continued the correct per-phone conversation style and reproduced the expected edit/rewrite behavior from the held-out source conversations.

W&B run:

https://wandb.ai/anothervibecoder-i-unemplyed/multiuser-phone-sft/runs/2l8h27ni

Note: `wandb sync` produced the run URL above, but reported a 403 while uploading `wandb-metadata.json`. Local W&B logs remain under `wandb/offline-run-20260618_061706-2l8h27ni`.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare interleaved data:

```bash
python prepare_data.py --out-dir data_realistic
```

Evaluate the base model:

```bash
python eval_model.py \
  --eval-file data_realistic/eval.jsonl \
  --out runs/baseline_realistic_eval.json
```

Fine-tune with W&B logging:

```bash
python train_lora.py \
  --train-file data_realistic/train.jsonl \
  --eval-file data_realistic/eval.jsonl \
  --out-dir runs/qwen-phone-lora-realistic \
  --wandb \
  --wandb-project multiuser-chat \
  --wandb-run-name qwen-phone-lora-realistic
```

If W&B is not logged in, the script automatically uses offline mode and writes local logs under `wandb/`.

Evaluate the adapter:

```bash
python eval_model.py \
  --eval-file data_realistic/eval.jsonl \
  --adapter runs/qwen-phone-lora-realistic \
  --out runs/finetuned_realistic_eval.json
```

Write a Markdown report:

```bash
python write_report.py \
  --baseline runs/baseline_realistic_eval.json \
  --finetuned runs/finetuned_realistic_eval.json \
  --trainer-metrics runs/qwen-phone-lora-realistic/final_metrics.json \
  --out runs/behavior_report_realistic.md
```

## Files

- `prepare_data.py`: filters WildChat and creates phone-prefix interleavings
- `train_lora.py`: LoRA SFT on the interleaved data
- `eval_model.py`: loss and generation comparison
- `write_report.py`: writes the base-versus-finetuned Markdown report
- `COMPARISON.md`: report from the completed run
