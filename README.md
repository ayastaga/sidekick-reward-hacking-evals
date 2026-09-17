# Reward hacking in a Sidekick-style GRPO loop
### Evaluator patching (Shopify's fix) vs gradient regularization (Ackermann et al., ICML 2026) — head to head

## The question

Shopify trains Sidekick with **GRPO against N-stage gated rewards** — a procedural validator (syntax, schema) gates an LLM judge — and has published the three ways the policy learned to cheat:

| Hack | What the model did | Why the reward liked it |
|---|---|---|
| Opt-out | "I'm unable to help with that" on a doable request | Judge rewarded honest-sounding replies |
| Tag-as-catch-all | `customer_tags CONTAINS 'enabled'` instead of `customer_account_status = 'ENABLED'` | Tags correlate with the real field; schema-blind judge can't tell |
| Schema violations | Invented IDs, invalid enums (`ACTIVATED`) | Validator only checked syntax (~93% accurate) |

Their published fix was **evaluator-side**: patch validators (93% → 99%) and judges (κ 0.66 → 0.75). Their 2026 post describes the mature loop (self-healing trajectories → SFT → GRPO with a calibrated judge) and says nothing about optimizer-side regularization.

Ackermann et al. (arXiv 2602.18037) propose an **optimizer-side** fix: reward hacking lives at sharp optima of the proxy reward; add (λ/2)‖∇L‖² so the policy stays in flat regions where the proxy is accurate. They show it stops a GRPO policy from hacking a Qwen2.5-1.5B judge, and lets a weaker judge match a stronger one.

Nobody has tested the two levers against each other on the failure modes Shopify actually reported. That's this project.

## Arms

| Arm | Evaluator | Regularizer | Reads as |
|---|---|---|---|
| `baseline` | weak (parse-only validator, schema-blind judge) | none | Shopify before their fix |
| `A` | patched (enum/field/ID checks, schema-aware anti-hack judge) | none | Shopify after their fix |
| `B` | weak | gradient regularization | paper's proposal, alone |
| `C` | patched | gradient regularization | both levers |
| `KL` | weak | KL penalty β=0.04 | the standard baseline the paper compares to |

Every arm is instrumented identically by detectors that **never enter the reward**: execution accuracy against a ground-truth customer database, plus opt-out / tag-hack / enum-hack / schema / junk rates.

## What's in here

```
segdsl/     dsl.py   — segment-filter language (parser is permissive by design; semantics live in validators)
            data.py  — synthetic store with *correlated tags* (the bait), NL→query tasks, 8% infeasible
rewards/    validators.py — weak vs patched stage-1 gates; hack detectors (measurement only)
            judge.py      — local LLM judge, weak vs patched prompts
            gated.py      — N-stage gated reward for TRL + per-step true-metric logging
training/   grad_reg.py   — GRPOTrainerGradReg: forward finite-difference GR, single GPU, no DeepSpeed
            sft.py, train_grpo.py, callbacks.py
evaluation/ evaluate.py, plot.py
tests/      env tests + numeric check of the GR estimator against exact H·g
run.ipynb   — Colab entry point (Drive-backed, resumable)
scripts/    run_all.py — one-command pipeline; evaluation/aggregate.py — mean ± std across seeds
```

## The GR implementation

Objective `J = L + (λ/2)‖∇L‖²`. Its gradient is `∇L + λ·H∇L`; we never form `H`. Following Karakida et al. (2023), which the paper uses:

```
g1 = ∇L(θ)
θ' = θ + ε · g1/‖g1‖
g2 = ∇L(θ')
∇J ≈ g1 + λ · ‖g1‖ · (g2 − g1)/ε
```

One extra backward per step on the *same* sampled completions (GRPO generates inside `_prepare_inputs`, so nothing is regenerated). Implemented by overriding `training_step` only, so it works on one GPU and with LoRA. The authors' reference fork patches TRL's `BaseTrainer` and Accelerate's DeepSpeed integration; ours is the same estimator without those dependencies, with the same knobs (`grad_reg_strength`, `grad_reg_eps`, warmup, g1/g2 clips). `tests/test_gr_math.py` checks the estimator against the exact Hessian-vector product on a quadratic.

## Reproduce

```
pip install -r requirements.txt && pytest tests
python scripts/run_all.py --quick            # ~15 min on an A100: pipeline check
python scripts/run_all.py --seeds 0 1 2      # ~4 h: 5 arms × 3 seeds + λ sweep on B → results/results.md
```

`run.ipynb` does the same from Colab with the working directory on Google Drive; every stage is resumable.

`run_all.py` builds the data, trains and picks an under-trained SFT warm start (exact match 0.70–0.85 so the policy still
explores), verifies with `tools/check_judge.py` that the weak reward is actually hackable, then chooses λ for GR by measuring
`gr/penalty_ratio` (‖λ·H∇L‖ / ‖∇L‖) over a few 4-step probes. λ is scale-dependent — the paper's default 1e-2 gave a ratio of
≈28 here, i.e. the curvature term would have erased the policy gradient — so arms B/C run at the λ whose ratio ≈ 0.3 and the
sweep covers ratios 0.1, 1, 3.

**Status of `figures/` and `runs/` currently in this repo:** single seed (plus one extra seed of arm A, which moved exec
score by 0.16), λ = 1e-4 chosen by hand, and a GR estimator that was biased by pre-clipping (fixed in `training/grad_reg.py`).
Treat them as a pilot. The numbers to report are whatever `results/results.md` says after the run above.

## How to read the result

- If `baseline` doesn't hack, the environment is too easy. Weaken the judge (smaller model), raise LR, or lengthen training. You need to reproduce the disease before testing cures.
- `A` should kill the hacks its validator can see. Watch **junk** and any drift the patched judge didn't anticipate — that residual is the case for an optimizer-side lever.
- `B` is the paper's claim in Shopify's setting: same weak evaluator, does GR alone hold true reward up? The paper predicts a gradient-norm spike at hacking onset in the unregularized run; check panel 6.
- `C > A` on exec score at equal hack rates → complementary. `C ≈ A` → patching sufficed here, and that's a real result too.
- `KL` vs `B`: the paper says GR beats KL at equal true reward. Verify.

## Honest scope

- 0.5B policy, 1.5B judge, synthetic store. Directional evidence, not a production claim.
- The DSL is modeled on Shopify's segment syntax, not their parser. The GraphQL agent (their 2026 flagship) is the natural next target and only touches `segdsl/` + validators.
- The judge prompts encode the *documented* hacks. A policy that finds an undocumented one is the interesting outcome, not a bug.

## References
- Shopify Engineering, *Building production-ready agentic systems: Lessons from Shopify Sidekick* (Aug 2025)
- Shopify Engineering, *Sidekick's continual learning loop* (Aug 2026)
- Ackermann, Noukhovitch, Ishida, Sugiyama. *Gradient Regularization Prevents Reward Hacking in RLHF and RLVR.* ICML 2026, arXiv:2602.18037. Code: JohannesAck/gradientregularization_trl
- Karakida et al. *Understanding Gradient Regularization in Deep Learning.* ICML 2023
- Thaman. *Reward Hacking Benchmark.* ICML 2026, arXiv:2605.02964 (measurement methodology)
