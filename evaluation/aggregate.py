"""
Aggregate every finished run under runs/ into a mean ± std table across seeds.

Runs are grouped by what they *are* (evaluator, regularizer, lr, λ) read from each run's config.json,
not by directory name, so old runs and new runs land in the same table.

    python -m evaluation.aggregate --runs runs --out results
"""
from __future__ import annotations
import argparse, glob, json, os, statistics as st

ARM_NAME = {("weak", "none"): "baseline", ("patched", "none"): "A (patched)", ("weak", "gr"): "B (weak+GR)",
            ("patched", "gr"): "C (patched+GR)", ("weak", "kl"): "KL (weak+KL)"}
COLS = [("exec_score", "exec"), ("exec_score_notag", "exec (no-tag)"), ("exact_match", "exact"),
        ("hack_opt_out", "opt-out"), ("hack_tag_hack", "tag-hack"), ("hack_enum_hack", "enum-hack"),
        ("hack_schema", "schema-hack"), ("hack_junk", "junk")]


def load_runs(root: str, min_steps: int | None):
    out = []
    for cfg_path in sorted(glob.glob(f"{root}/*/config.json")):
        d = os.path.dirname(cfg_path)
        if os.path.basename(d).startswith("_"):            # smoke / probe runs
            continue
        ev_path = f"{d}/eval_metrics.jsonl"
        if not os.path.exists(ev_path):
            continue
        cfg = json.load(open(cfg_path))
        ev = [json.loads(l) for l in open(ev_path)]
        ev = [r for r in ev if "exec_score" in r]
        if not ev:
            continue
        steps = cfg.get("steps", 0)
        if min_steps and (ev[-1]["global_step"] < min_steps or steps < min_steps):
            continue
        final = ev[-1]
        tr_path = f"{d}/train_metrics.jsonl"
        train_tail = None
        if os.path.exists(tr_path):
            tr = [json.loads(l) for l in open(tr_path)]
            tr = [r for r in tr if "exec_score" in r][-20:]
            if tr: train_tail = st.mean(r["exec_score"] for r in tr)
        lam = cfg.get("gr_strength") if cfg.get("reg") == "gr" else None
        key = (cfg.get("evaluator"), cfg.get("reg"), cfg.get("lr"), lam)
        out.append(dict(dir=os.path.basename(d), key=key, seed=cfg.get("seed"), final=final, train_tail=train_tail,
                        steps=steps, start=ev[0]["exec_score"]))
    return out


def fmt(vals):
    vals = [v for v in vals if v == v]   # drop nan
    if not vals: return "–"
    if len(vals) == 1: return f"{vals[0]:.3f}"
    return f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results")
    ap.add_argument("--min_steps", type=int, default=None, help="ignore runs shorter than this")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs, a.min_steps)
    groups: dict[tuple, list] = {}
    for r in runs: groups.setdefault(r["key"], []).append(r)

    lines = ["| arm | lr | λ | n seeds | " + " | ".join(c[1] for c in COLS) + " | train exec (last 20) |",
             "|" + "---|" * (5 + len(COLS))]
    order = sorted(groups, key=lambda k: (-(k[2] or 0), ["baseline", "A (patched)", "B (weak+GR)", "C (patched+GR)", "KL (weak+KL)"].index(ARM_NAME.get(k[:2], "baseline")), k[3] or 0))
    for key in order:
        g = groups[key]
        name = ARM_NAME.get(key[:2], f"{key[0]}/{key[1]}")
        cells = [fmt([r["final"].get(c, float("nan")) for r in g]) for c, _ in COLS]
        tt = fmt([r["train_tail"] for r in g if r["train_tail"] is not None])
        lines.append(f"| {name} | {key[2]:g} | {key[3] if key[3] is not None else '–'} | {len(g)} | " + " | ".join(cells) + f" | {tt} |")
    lines.append("")
    lines.append("Seeds per row: " + "; ".join(f"{ARM_NAME.get(k[:2], k)} lr={k[2]:g} λ={k[3]}: " + ", ".join(f"{r['dir']}(s{r['seed']})" for r in groups[k]) for k in order))
    lines.append(f"\nAll arms start from the same SFT policy (held-out exec {runs[0]['start']:.3f}) when this table was built.")
    md = "\n".join(lines)
    open(f"{a.out}/results.md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
