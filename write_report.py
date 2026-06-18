import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_float(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def sample_map(metrics: dict) -> dict:
    return {s["context_last_user"]: s for s in metrics.get("samples", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--trainer-metrics", default=None)
    parser.add_argument("--out", default="runs/behavior_report.md")
    parser.add_argument("--title", default="Phone-Prefix Multiuser SFT Behavior Report")
    args = parser.parse_args()

    base = load_json(args.baseline)
    tuned = load_json(args.finetuned)
    trainer = load_json(args.trainer_metrics) if args.trainer_metrics and Path(args.trainer_metrics).exists() else {}

    base_samples = sample_map(base)
    tuned_samples = sample_map(tuned)
    common = [k for k in base_samples if k in tuned_samples]

    lines = [
        f"# {args.title}",
        "",
        "## Setup",
        "",
        "- Base model: `Qwen/Qwen2.5-0.5B-Instruct`",
        "- Fine-tuning method: LoRA SFT on interleaved WildChat conversations",
        "- User messages use realistic-looking phone-number prefixes.",
        "- Assistant targets do not include phone prefixes or chain-of-thought.",
        "",
        "## Metrics",
        "",
        "| Metric | Before fine-tune | After fine-tune |",
        "| --- | ---: | ---: |",
        f"| Eval loss | {fmt_float(base.get('eval_loss'))} | {fmt_float(tuned.get('eval_loss'))} |",
        f"| Perplexity | {fmt_float(base.get('perplexity'))} | {fmt_float(tuned.get('perplexity'))} |",
        f"| Mean word F1 | {fmt_float(base.get('mean_word_f1'))} | {fmt_float(tuned.get('mean_word_f1'))} |",
        f"| Phone-prefix violation rate | {fmt_float(base.get('phone_prefix_violation_rate'))} | {fmt_float(tuned.get('phone_prefix_violation_rate'))} |",
        "",
    ]

    if trainer:
        lines += [
            "## Trainer Summary",
            "",
            f"- Final trainer eval loss: `{fmt_float(trainer.get('eval_loss'))}`",
            f"- Final trainer eval runtime seconds: `{fmt_float(trainer.get('eval_runtime'))}`",
            "",
        ]

    lines += [
        "## Behavioral Samples",
        "",
        "These are held-out generation cases. The reference is the original assistant reply from the source conversation.",
        "",
    ]

    for idx, key in enumerate(common[:6], start=1):
        b = base_samples[key]
        t = tuned_samples[key]
        lines += [
            f"### Sample {idx}",
            "",
            "**Latest user message**",
            "",
            "```text",
            key,
            "```",
            "",
            "**Reference**",
            "",
            "```text",
            b["reference"],
            "```",
            "",
            f"**Before fine-tune** (`word_f1={fmt_float(b.get('word_f1'))}`)",
            "",
            "```text",
            b["prediction"],
            "```",
            "",
            f"**After fine-tune** (`word_f1={fmt_float(t.get('word_f1'))}`)",
            "",
            "```text",
            t["prediction"],
            "```",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- Lower eval loss means the model assigns higher likelihood to the held-out assistant responses.",
        "- Word F1 is a rough lexical proxy, not a full quality metric.",
        "- Phone-prefix violation checks whether the assistant incorrectly starts its answer with a phone number.",
        "- The comparison is only meaningful for the exact held-out split used in this run.",
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
