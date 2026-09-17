"""
The paper's headline figure, for our arms: proxy (judge) reward vs true (execution) reward over training,
plus hack-rate panels and gradient norm.

    python -m evaluation.plot runs/baseline runs/A_patched runs/B_weak_gr runs/C_patched_gr --out figures
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

def smooth(xs, k=5):
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - k + 1): i + 1]; out.append(sum(w) / len(w))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    panels = [("proxy_reward", "Proxy reward (gated judge)", "train"), ("exec_score", "True reward (execution Jaccard, held-out)", "eval"),
              ("hack_opt_out", "Opt-out rate (refused feasible)", "eval"), ("hack_tag_hack", "Tag-hack rate", "eval"),
              ("hack_enum_hack", "Invalid-enum rate", "eval"), ("gr/g1_norm", "Gradient norm ||g1||", "train")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        tr, ev = read(f"{run}/train_metrics.jsonl"), read(f"{run}/eval_metrics.jsonl")
        for ax, (key, title, src) in zip(axes.flat, panels):
            rows = tr if src == "train" else ev
            xs = [r["global_step"] for r in rows if key in r]; ys = [r[key] for r in rows if key in r]
            if key == "gr/g1_norm" and not ys:  # arms without GR: fall back to trainer grad_norm
                xs = [r["global_step"] for r in tr if "grad_norm" in r]; ys = [r["grad_norm"] for r in tr if "grad_norm" in r]
            if ys: ax.plot(xs, smooth(ys) if src == "train" else ys, label=name)
            ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=.3)
    axes.flat[0].legend()
    fig.suptitle("Reward hacking under GRPO: evaluator patching (A) vs gradient regularization (B) vs both (C)")
    fig.tight_layout(); fig.savefig(f"{a.out}/summary.png", dpi=140); print("wrote", f"{a.out}/summary.png")

    # Final-eval table
    rows = []
    for run in a.runs:
        ev = read(f"{run}/eval_metrics.jsonl")
        if ev:
            r = ev[-1]; rows.append((os.path.basename(run.rstrip("/")), r["exec_score"], r.get("exec_score_notag", float("nan")), r.get("exec_score_tag", float("nan")), r["exact_match"], r["hack_opt_out"], r["hack_tag_hack"], r["hack_enum_hack"], r["hack_junk"]))
    with open(f"{a.out}/final_table.md", "w") as f:
        f.write("| arm | exec | exec (no-tag tasks) | exec (tag tasks) | exact | opt-out | tag-hack | enum-hack | junk |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in rows: f.write("| " + " | ".join([r[0]] + [f"{x:.3f}" for x in r[1:]]) + " |\n")
    print(open(f"{a.out}/final_table.md").read())


if __name__ == "__main__":
    main()
