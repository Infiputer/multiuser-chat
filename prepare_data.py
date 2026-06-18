import json
import argparse
import random
import re
from pathlib import Path

from datasets import load_dataset


SYSTEM = (
    "You handle messages from multiple users. Each user message begins with a phone number. "
    "Reply only to the latest user message. Do not include the phone number in your reply. "
    "Keep each phone number's context separate. If the latest phone number has not provided "
    "enough information, ask a concise follow-up question or say you do not know. Do not reveal reasoning."
)

BAD_PATTERNS = [
    "```",
    "write a very long",
    "shooting script",
    "screenplay",
    "essay",
    "python",
    "javascript",
    "sql",
    "latex",
    "files",
    "block of text",
    "summarize the following",
    "translate",
    "act as",
    "ignore previous",
    "base64",
    "step by step",
    "let's solve",
    "mathematic formula",
    "which of the following",
    "derive",
    "prove that",
    "chain of thought",
    "show your reasoning",
]

PERSONAL_MARKERS = [
    " i ",
    " my ",
    " me ",
    " friend",
    " mom",
    " dad",
    " sister",
    " brother",
    " school",
    " class",
    " work",
    " boss",
    " birthday",
    " feel",
    " advice",
    " plan",
    " help me",
    " should i",
    " make it",
    " change it",
]

FOLLOWUP_MARKERS = [
    "yes",
    "no",
    "ok",
    "okay",
    "same",
    "shorter",
    "longer",
    "more",
    "less",
    "that",
    "this",
    "it",
    "them",
    "make it",
    "change it",
    "what about",
    "can you",
]


def word_count(text: str) -> int:
    return len((text or "").split())


def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def mostly_plain_english(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    return ascii_chars / max(len(text), 1) >= 0.92 and alpha_chars >= 10


def is_good_conversation(row: dict) -> bool:
    conv = row.get("conversation") or []
    if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
        return False
    if not (4 <= len(conv) <= 8) or len(conv) % 2 != 0:
        return False
    if any(m.get("role") not in {"user", "assistant"} for m in conv):
        return False
    if any(conv[i].get("role") == conv[i + 1].get("role") for i in range(len(conv) - 1)):
        return False

    texts = [clean_text(m.get("content", "")) for m in conv]
    if not all(mostly_plain_english(t) for t in texts):
        return False
    lens = [word_count(t) for t in texts]
    joined = f" {' '.join(texts).lower()} "
    if max(lens) > 120 or sum(lens) > 520:
        return False
    if any(bad in joined for bad in BAD_PATTERNS):
        return False
    if "as an ai language model" in joined:
        return False

    user_texts = [texts[i].lower() for i in range(0, len(texts), 2)]
    later_users = user_texts[1:]
    has_dependency = any(
        word_count(t) <= 12 or any(marker in f" {t} " for marker in FOLLOWUP_MARKERS)
        for t in later_users
    )
    personal_score = sum(marker in joined for marker in PERSONAL_MARKERS)
    return has_dependency and personal_score >= 1


def row_to_turns(row: dict) -> list[tuple[str, str]]:
    conv = row["conversation"]
    turns = []
    for i in range(0, len(conv), 2):
        turns.append((clean_text(conv[i]["content"]), clean_text(conv[i + 1]["content"])))
    return turns


BLOCKED_N11 = {211, 311, 411, 511, 611, 711, 811, 911}


def phone_number(rng: random.Random, used: set[str]) -> str:
    while True:
        area = rng.randint(201, 989)
        exchange = rng.randint(201, 989)
        if area in BLOCKED_N11 or exchange in BLOCKED_N11 or exchange == 555:
            continue
        phone = f"+1-{area}-{exchange}-{rng.randint(1000, 9999)}"
        if phone not in used:
            used.add(phone)
            return phone


def make_interleaved(conversations: list[list[tuple[str, str]]], rng: random.Random) -> dict:
    used: set[str] = set()
    queues = []
    for turns in conversations:
        queues.append({"phone": phone_number(rng, used), "turns": turns, "idx": 0})

    messages = [{"role": "system", "content": SYSTEM}]
    active = queues[:]
    while active:
        q = rng.choice(active)
        user_text, assistant_text = q["turns"][q["idx"]]
        messages.append({"role": "user", "content": f"{q['phone']}: {user_text}"})
        messages.append({"role": "assistant", "content": assistant_text})
        q["idx"] += 1
        active = [item for item in active if item["idx"] < len(item["turns"])]
    return {"messages": messages}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--source-conversations", type=int, default=260)
    parser.add_argument("--train-examples", type=int, default=180)
    parser.add_argument("--eval-examples", type=int, default=40)
    parser.add_argument("--max-scan", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = load_dataset("allenai/WildChat", split="train", streaming=True).shuffle(
        seed=args.seed, buffer_size=10_000
    )

    source = []
    scanned = 0
    for row in dataset:
        scanned += 1
        if is_good_conversation(row):
            source.append(row_to_turns(row))
        if len(source) >= args.source_conversations or scanned >= args.max_scan:
            break

    min_required = min(args.source_conversations, 20)
    if len(source) < min_required:
        raise RuntimeError(f"Only found {len(source)} usable conversations after scanning {scanned} rows")

    examples = []
    total = args.train_examples + args.eval_examples
    for _ in range(total):
        n = rng.randint(2, 4)
        picked = [rng.choice(source) for _ in range(n)]
        examples.append(make_interleaved(picked, rng))

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", examples[: args.train_examples])
    write_jsonl(out_dir / "eval.jsonl", examples[args.train_examples :])
    write_jsonl(out_dir / "source_sample.jsonl", [{"turns": s} for s in source[:20]])

    print(
        json.dumps(
            {
                "scanned": scanned,
                "source_conversations": len(source),
                "train_examples": args.train_examples,
                "eval_examples": args.eval_examples,
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
