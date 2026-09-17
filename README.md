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
sidekick_reward_hacking.ipynb — the Colab walkthrough
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

## Walkthrough (Colab)

1. **Install** pinned TRL/transformers (the fork's tested versions; our override uses the same `training_step` signature).
2. **Tests** — `pytest tests` should pass on CPU before you spend GPU time.
3. **Build data** — `python -m segdsl.data data`. Check the bait: `ENABLED` vs `tags CONTAINS 'enabled'` Jaccard ≈ 0.58.
4. **SFT** — 2 epochs on 800 gold pairs so the 0.5B model knows the syntax. Eval should show high exec score and ~0 hacks; this is the starting point every arm shares.
5. **Smoke test** — 2 GRPO steps on arm B. Confirms `gr/g1_norm` appears in `train_metrics.jsonl`.
6. **Run arms** — 300 steps each, seed 0. ~30 min/arm on A100.
7. **Plot** — `evaluation/plot.py` gives the paper's proxy-vs-true figure plus one panel per Shopify hack.
8. **Then**: sweep λ ∈ {1e-3, 1e-2, 1e-1} on B; three seeds on whatever you report.

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
