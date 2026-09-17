"""
Evaluate every SFT checkpoint and pick the warm start with the right amount of competence.

Target band: exact match 0.70-0.85. Above that the policy is deterministic (entropy collapses, GRPO
groups have no variance, nothing to explore). Below it, the model is still inventing field names and
GRPO relearns syntax instead of finding exploits.

    python -m tools.pick_sft --run runs/sft --data data --n 200
    python -m tools.pick_sft --run runs/sft --promote checkpoint-40   # then copy it into place
"""
from __future__ import annotations
import argparse, glob, json, os, re, shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.evaluate import evaluate_model, load_tasks
from segdsl.data import load_customers

LO, HI = 0.70, 0.85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/sft"); ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--promote", default=None)
    a = ap.parse_args()

    if a.promote:
        src = os.path.join(a.run, a.promote)
        dst = os.path.join(a.run, "final")
        shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
        print(f"promoted {src} -> {dst}"); return

    tasks = load_tasks(f"{a.data}/test.jsonl")[: a.n]
    customers = load_customers(f"{a.data}/customers.json")
    cks = sorted(glob.glob(f"{a.run}/checkpoint-*"), key=lambda p: int(re.search(r"\d+$", p).group()))
    cks.append(f"{a.run}/final")

    print(f"{'checkpoint':>22} | {'exact':>6} {'exec':>6} {'refuse':>6} {'schema':>6} {'len':>5}")
    print("-" * 64)
    rows = []
    for ck in cks:
        tok = AutoTokenizer.from_pretrained(ck)
        model = AutoModelForCausalLM.from_pretrained(ck, dtype=torch.bfloat16).cuda().eval()
        r = evaluate_model(model, tok, tasks, customers)
        rows.append((ck, r))
        flag = "  <-- in band" if LO <= r["exact_match"] <= HI else ""
        print(f"{os.path.basename(ck):>22} | {r['exact_match']:6.3f} {r['exec_score']:6.3f} "
              f"{r['refusal']:6.3f} {r['hack_schema']:6.3f} {r['completion_len']:5.0f}{flag}")
        del model; torch.cuda.empty_cache()

    in_band = [(c, r) for c, r in rows if LO <= r["exact_match"] <= HI]
    if in_band:
        best = min(in_band, key=lambda cr: abs(cr[1]["exact_match"] - 0.78))
        print(f"\nPick: {best[0]}  (exact {best[1]['exact_match']:.3f})")
        print(f"Run:  python -m tools.pick_sft --run {a.run} --promote {os.path.basename(best[0])}")
    else:
        hi = max(rows, key=lambda cr: cr[1]["exact_match"])
        print(f"\nNo checkpoint in [{LO}, {HI}]. Best is {os.path.basename(hi[0])} at {hi[1]['exact_match']:.3f}.")
        print("If all are too LOW  : rerun sft.py with --epochs 3 or --lr 2e-5.")
        print("If all are too HIGH : rerun with --save_steps 5 to get finer-grained early checkpoints.")


if __name__ == "__main__":
    main()
