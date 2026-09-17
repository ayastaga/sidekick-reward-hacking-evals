"""
One-command, resumable pipeline. Every stage checks for its own output and skips if done, so after a
Colab disconnect you just run the same command again. Keep the working directory on Google Drive.

    python scripts/run_all.py --seeds 0 1 2 --steps 300

Stages
  1. data          segdsl.data  → data/*.jsonl
  2. sft           training.sft + tools.pick_sft → runs/sft/final (weights kept; this is what was lost last time)
  3. judge check   tools.check_judge → results/check_judge.txt  (proves the weak reward is actually hackable)
  4. λ probe       4-step GR runs at several λ → results/gr_probe.json; λ is chosen by gr/penalty_ratio,
                   because λ is scale-dependent (the paper's 1e-2 gave ratio ≈ 28 here, i.e. no policy learning)
  5. main arms     baseline / A / B / C / KL × seeds, at the λ whose ratio ≈ --target_ratio
  6. λ sweep       arm B, seed 0, at the λ's matching each --sweep_ratios entry
  7. aggregate     evaluation.aggregate → results/results.md ; evaluation.plot → results/figures (seed 0)

Fast sanity pass (≈15 min on an A100): --quick   (2 seeds? no: 1 seed, 60 steps, no sweep)
"""
from __future__ import annotations
import argparse, json, math, os, shutil, statistics as st, subprocess, sys, time

ARMS = ["baseline", "A", "B", "C", "KL"]
PY = sys.executable


def sh(cmd: str, log: str | None = None):
    print(f"\n$ {cmd}", flush=True)
    if log:
        with open(log, "w") as f:
            p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {cmd}")


def run_done(out: str, steps: int) -> bool:
    ev = f"{out}/eval_metrics.jsonl"
    if not os.path.exists(ev): return False
    rows = [json.loads(l) for l in open(ev)]
    return any(r.get("global_step", -1) >= steps for r in rows)


def grpo(arm: str, out: str, a, seed: int, lam: float | None, steps: int | None = None, extra: str = ""):
    steps = steps or a.steps
    if run_done(out, steps):
        print(f"[skip] {out} already finished"); return
    shutil.rmtree(out, ignore_errors=True)          # partial run: restart clean, never append
    lam_arg = f"--gr_strength {lam}" if lam is not None else ""
    sh(f"{PY} -m training.train_grpo --arm {arm} --policy runs/sft/final --data data --out {out} "
       f"--steps {steps} --seed {seed} --lr {a.lr} --batch {a.batch} --num_generations {a.num_generations} "
       f"--temperature {a.temperature} --eval_every {a.eval_every} --eval_n {a.eval_n} {lam_arg} {extra}",
       log=f"{out}.log" if a.quiet else None)


def probe_ratio(lam: float, a) -> float:
    out = "runs/_probe"
    shutil.rmtree(out, ignore_errors=True)
    sh(f"{PY} -m training.train_grpo --arm B --policy runs/sft/final --data data --out {out} --steps 4 "
       f"--eval_every 999 --eval_n 8 --lr {a.lr} --batch {a.batch} --num_generations {a.num_generations} "
       f"--temperature {a.temperature} --gr_strength {lam} --gr_max_ratio 0", log=f"{out}.log")
    rows = [json.loads(l) for l in open(f"{out}/train_metrics.jsonl")]
    rs = [r["gr/penalty_ratio"] for r in rows if "gr/penalty_ratio" in r]
    shutil.rmtree(out, ignore_errors=True)
    return st.mean(rs) if rs else float("nan")


def pick_lambda(probe: dict[str, float], target: float) -> float:
    """λ whose measured ratio is nearest to target in log space; interpolate between neighbours."""
    pts = sorted((float(k), v) for k, v in probe.items() if v == v and v > 0)
    if not pts: raise SystemExit("λ probe produced no ratios")
    lt = math.log(target)
    for (l0, r0), (l1, r1) in zip(pts, pts[1:]):
        if math.log(r0) <= lt <= math.log(r1):
            t = (lt - math.log(r0)) / (math.log(r1) - math.log(r0) + 1e-12)
            return math.exp(math.log(l0) + t * (math.log(l1) - math.log(l0)))
    return min(pts, key=lambda p: abs(math.log(p[1]) - lt))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5, help="1e-6 did nothing in 300 steps; 1e-5 collapses the weak arm in ~50")
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--eval_every", type=int, default=25); ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--target_ratio", type=float, default=0.3, help="gr/penalty_ratio for the main B/C arms")
    ap.add_argument("--sweep_ratios", type=float, nargs="*", default=[0.1, 1.0, 3.0], help="extra ratios for the arm-B λ sweep")
    ap.add_argument("--probe_lams", type=float, nargs="*", default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--sft_n", type=int, default=800); ap.add_argument("--sft_epochs", type=float, default=2)
    ap.add_argument("--quick", action="store_true", help="1 seed, 60 steps, no sweep — verifies the pipeline end to end")
    ap.add_argument("--quiet", action="store_true", help="send training stdout to <run>.log instead of the console")
    ap.add_argument("--skip_sweep", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.steps, a.eval_every, a.eval_n, a.skip_sweep = [0], 60, 20, 50, True
    os.makedirs("results", exist_ok=True); os.makedirs("runs", exist_ok=True)
    t0 = time.time()

    # 1. data
    if not os.path.exists("data/train.jsonl"):
        sh(f"{PY} -m segdsl.data data")

    # 2. SFT warm start (weights persist under runs/sft/final)
    if not os.path.exists("runs/sft/final/model.safetensors"):
        shutil.rmtree("runs/sft", ignore_errors=True)
        sh(f"{PY} -m training.sft --model Qwen/Qwen2.5-0.5B-Instruct --data data --out runs/sft "
           f"--n {a.sft_n} --epochs {a.sft_epochs} --lr 1e-5 --save_steps 10")
        sh(f"{PY} -m tools.pick_sft --run runs/sft --data data --n 200", log="results/sft_pick.txt")
        txt = open("results/sft_pick.txt").read()
        line = [l for l in txt.splitlines() if l.startswith("Pick:")]
        if not line: raise SystemExit("no SFT checkpoint landed in the 0.70–0.85 band; see results/sft_pick.txt")
        ck = os.path.basename(line[0].split()[1])
        sh(f"{PY} -m tools.pick_sft --run runs/sft --promote {ck}")
        sh("rm -rf runs/sft/checkpoint-*")
        sh(f"{PY} -m evaluation.evaluate --model runs/sft/final --data data --n 400", log="results/sft_eval.json")
    print("SFT policy ready:", open("results/sft_pick.txt").read().splitlines()[-2] if os.path.exists("results/sft_pick.txt") else "(pre-existing)")

    # 3. is the reward hackable / is the patch effective? (keep the evidence)
    if not os.path.exists("results/check_judge.txt"):
        sh(f"{PY} -m tools.check_judge --judge Qwen/Qwen2.5-1.5B-Instruct", log="results/check_judge.txt")
    fails = [l for l in open("results/check_judge.txt") if "[FAIL]" in l]
    if fails: print("check_judge FAILs (read results/check_judge.txt before trusting arm B/C):\n" + "".join(fails))

    # 4. λ probe → λ by ratio
    probe_path = "results/gr_probe.json"
    if os.path.exists(probe_path):
        probe = json.load(open(probe_path))
    else:
        probe = {}
        for lam in a.probe_lams:
            probe[str(lam)] = probe_ratio(lam, a)
            print(f"  λ={lam:.0e}  penalty/gradient ratio = {probe[str(lam)]:.3f}")
        json.dump(probe, open(probe_path, "w"), indent=2)
    lam_main = pick_lambda(probe, a.target_ratio)
    print(f"λ for ratio≈{a.target_ratio}: {lam_main:.2e}")

    # 5. main arms × seeds
    for seed in a.seeds:
        for arm in ARMS:
            grpo(arm, f"runs/{arm}_s{seed}", a, seed, lam_main if arm in ("B", "C") else None)

    # 6. λ sweep on B (seed 0)
    if not a.skip_sweep:
        for r in a.sweep_ratios:
            lam = pick_lambda(probe, r)
            grpo("B", f"runs/B_ratio{r:g}_s0", a, 0, lam)

    # 7. aggregate + plot
    sh(f"{PY} -m evaluation.aggregate --runs runs --out results --min_steps {a.steps}")
    seed0 = " ".join(f"runs/{arm}_s{a.seeds[0]}" for arm in ARMS if run_done(f"runs/{arm}_s{a.seeds[0]}", a.steps))
    sh(f"{PY} -m evaluation.plot {seed0} --out results/figures")
    print(f"\nall done in {(time.time() - t0) / 60:.0f} min → results/results.md, results/figures/summary.png")


if __name__ == "__main__":
    main()
