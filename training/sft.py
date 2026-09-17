"""
Stage 0: supervised warm start. Shopify distills healed trajectories with SFT before GRPO; we do the
cheap equivalent — gold (prompt → query) pairs so the base model knows the syntax.

We do NOT guess how much SFT is right. Train the full recipe, checkpoint along the way, then pick
the checkpoint whose exact match lands in the target band (tools/pick_sft.py). Too little SFT and
GRPO spends its budget relearning syntax; too much and the policy is deterministic with nothing to
explore — and reward hacking is a search phenomenon.

    python -m training.sft --model Qwen/Qwen2.5-0.5B-Instruct --data data --out runs/sft
"""
from __future__ import annotations
import argparse, glob, json, os, shutil
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from segdsl.data import SYSTEM_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n", type=int, default=800, help="subsample SFT pairs")
    ap.add_argument("--save_steps", type=int, default=10, help="checkpoint cadence for the sweep")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{a.data}/sft.jsonl")][: a.n]
    ds = Dataset.from_list([{"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r["gold"] if r["feasible"] else "UNSUPPORTED"},
    ]} for r in rows])

    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=8,
                    gradient_accumulation_steps=2, bf16=True, logging_steps=5, save_strategy="steps",
                    save_steps=a.save_steps, save_total_limit=None, report_to="none",
                    max_length=512, assistant_only_loss=False)
    trainer = SFTTrainer(model=a.model, args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(f"{a.out}/final"); trainer.processing_class.save_pretrained(f"{a.out}/final")

    # checkpoints don't always carry tokenizer files; copy so each is independently loadable
    for ck in glob.glob(f"{a.out}/checkpoint-*"):
        for f in (glob.glob(f"{a.out}/final/*token*") + glob.glob(f"{a.out}/final/vocab.json")
                  + glob.glob(f"{a.out}/final/merges.txt")):
            dst = os.path.join(ck, os.path.basename(f))
            if not os.path.exists(dst): shutil.copy(f, dst)
    print("saved", f"{a.out}/final", "and", len(glob.glob(f"{a.out}/checkpoint-*")), "checkpoints")


if __name__ == "__main__":
    main()
