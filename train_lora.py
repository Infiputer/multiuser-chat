import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


class ChatDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def _encode_messages(self, messages):
        input_ids = []
        labels = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                header = "<|im_start|>system\n"
            elif role == "user":
                header = "<|im_start|>user\n"
            elif role == "assistant":
                header = "<|im_start|>assistant\n"
            else:
                raise ValueError(role)
            footer = "<|im_end|>\n"

            header_ids = self.tokenizer(header, add_special_tokens=False).input_ids
            content_ids = self.tokenizer(content, add_special_tokens=False).input_ids
            footer_ids = self.tokenizer(footer, add_special_tokens=False).input_ids

            input_ids.extend(header_ids)
            labels.extend([-100] * len(header_ids))
            input_ids.extend(content_ids)
            labels.extend(content_ids if role == "assistant" else [-100] * len(content_ids))
            input_ids.extend(footer_ids)
            labels.extend(footer_ids if role == "assistant" else [-100] * len(footer_ids))

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
            labels = labels[-self.max_length :]
            if all(x == -100 for x in labels):
                labels[-1] = input_ids[-1]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __getitem__(self, idx):
        return self._encode_messages(self.rows[idx]["messages"])


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--out-dir", default="runs/qwen-phone-lora")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="multiuser-phone-sft")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline"], default="auto")
    args = parser.parse_args()

    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        if args.wandb_mode == "offline" or (args.wandb_mode == "auto" and not os.environ.get("WANDB_API_KEY")):
            os.environ.setdefault("WANDB_MODE", "offline")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = ChatDataset(args.train_file, tokenizer, args.max_length)
    eval_ds = ChatDataset(args.eval_file, tokenizer, args.max_length)
    collator = DataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=["wandb"] if args.wandb else [],
        run_name=args.wandb_run_name,
        optim="paged_adamw_8bit" if quant is not None else "adamw_torch",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    metrics = trainer.evaluate()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
