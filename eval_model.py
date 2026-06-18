import argparse
import json
import math
import re
from pathlib import Path

import torch
from peft import PeftModel
from train_lora import ChatDataset, DataCollator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PHONE_RE = re.compile(r"^\s*\+?1[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}\s*:")


def word_f1(pred: str, ref: str) -> float:
    p = re.findall(r"\w+", pred.lower())
    r = re.findall(r"\w+", ref.lower())
    if not p or not r:
        return 0.0
    common = {}
    for w in p:
        common[w] = common.get(w, 0) + 1
    overlap = 0
    for w in r:
        if common.get(w, 0) > 0:
            overlap += 1
            common[w] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def load_rows(path: str):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def eval_loss(model, tokenizer, path: str, max_length: int, limit: int):
    ds = ChatDataset(path, tokenizer, max_length)
    collator = DataCollator(tokenizer)
    losses = []
    model.eval()
    with torch.no_grad():
        for i in range(min(limit, len(ds))):
            batch = collator([ds[i]])
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.detach().cpu()))
    mean = sum(losses) / len(losses)
    return {"eval_loss": mean, "perplexity": math.exp(min(mean, 20))}


def generation_cases(rows, limit):
    cases = []
    for row in rows:
        msgs = row["messages"]
        for idx, msg in enumerate(msgs):
            if msg["role"] == "assistant" and idx > 0:
                cases.append((msgs[:idx], msg["content"]))
            if len(cases) >= limit:
                return cases
    return cases


def generate(model, tokenizer, context):
    prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, inputs.input_ids.shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--out", default="runs/eval.json")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--loss-limit", type=int, default=40)
    parser.add_argument("--gen-limit", type=int, default=30)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    rows = load_rows(args.eval_file)
    metrics = eval_loss(model, tokenizer, args.eval_file, args.max_length, args.loss_limit)
    cases = generation_cases(rows, args.gen_limit)
    outputs = []
    f1s = []
    prefix_violations = 0
    for context, ref in cases:
        pred = generate(model, tokenizer, context)
        f1 = word_f1(pred, ref)
        f1s.append(f1)
        prefix_violations += int(bool(PHONE_RE.match(pred)))
        outputs.append({"context_last_user": context[-1]["content"], "reference": ref, "prediction": pred, "word_f1": f1})

    metrics.update(
        {
            "generation_cases": len(cases),
            "mean_word_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "phone_prefix_violations": prefix_violations,
            "phone_prefix_violation_rate": prefix_violations / len(cases) if cases else 0.0,
            "samples": outputs[:10],
        }
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
