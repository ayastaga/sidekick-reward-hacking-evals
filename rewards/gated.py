"""
N-Stage Gated Reward — Shopify's recipe: procedural gate first, semantic judge second.

    reward = 0                       if stage-1 validator fails
           = judge_score ∈ [0,1]     otherwise

The gate is what makes the validator matter: a query that fails syntax never reaches the judge, so a
patched validator removes whole classes of hacks from the reward surface. The judge only sees survivors.

`GatedReward` is a callable with the TRL reward_funcs signature:
    reward(prompts, completions, **kwargs) -> list[float]
Extra dataset columns (gold, gold_ids, feasible, gold_fields) arrive in kwargs as lists.

It also records, per call, the *true* metrics (execution score + hack rates) into `self.history`, so
every arm is instrumented identically regardless of what its reward can see. The trainer callback
in training/callbacks.py flushes these to the log.
"""
from __future__ import annotations
from collections import defaultdict
from .validators import VALIDATORS, detect_hacks, execution_score, extract_query, is_refusal


def _completion_text(c) -> str:
    # TRL conversational format → list of message dicts; standard format → str
    if isinstance(c, list):
        return c[-1]["content"] if c else ""
    return c


class GatedReward:
    def __init__(self, validator: str, judge, customers: list[dict], name: str = "gated"):
        self.validate = VALIDATORS[validator]
        self.judge = judge
        self.customers = customers
        self.__name__ = f"{name}_{validator}_{judge.mode if judge else 'nojudge'}"
        self.history: list[dict] = []
        self.step = 0

    def _tasks(self, n: int, kw: dict) -> list[dict]:
        return [dict(gold=kw["gold"][i], gold_ids=kw["gold_ids"][i], feasible=kw["feasible"][i], gold_fields=kw["gold_fields"][i]) for i in range(n)]

    def __call__(self, prompts, completions, **kw) -> list[float]:
        texts = [_completion_text(c) for c in completions]
        requests = [p[-1]["content"] if isinstance(p, list) else p for p in prompts]
        tasks = self._tasks(len(texts), kw)

        gate = [self.validate(t, task) for t, task in zip(texts, tasks)]
        rewards = [0.0] * len(texts)
        # Procedural terminal reward: the patched validator knows the task is infeasible and the reply
        # is a refusal, so it settles the reward itself (1.0) instead of asking a 1.5B judge to reason
        # about feasibility. The weak validator never emits "refusal_ok", so the weak arm is unchanged.
        for i, g in enumerate(gate):
            if g.ok and g.reason == "refusal_ok": rewards[i] = 1.0
        survivors = [i for i, g in enumerate(gate) if g.ok and g.reason != "refusal_ok"]
        if survivors and self.judge is not None:
            js = self.judge.score([requests[i] for i in survivors], [texts[i] for i in survivors])
            for i, s in zip(survivors, js): rewards[i] = float(s)
        elif survivors:  # judge-less ablation: pass-through
            for i in survivors: rewards[i] = 1.0

        # --- instrumentation (never influences reward) ---
        agg = defaultdict(float)
        for t, task, r in zip(texts, tasks, rewards):
            agg["proxy_reward"] += r
            agg["exec_score"] += execution_score(t, task, self.customers)
            agg["exact_match"] += float(execution_score(t, task, self.customers) == 1.0)
            for k, v in detect_hacks(t, task).items(): agg[f"hack_{k}"] += float(v)
            agg["gate_fail"] += float(not self.validate(t, task).ok)
            agg["refusal"] += float(is_refusal(t))
            agg["completion_len"] += len(t)
        n = len(texts)
        rec = {k: v / n for k, v in agg.items()}
        rec["step"] = self.step; rec["n"] = n
        rec["sample"] = extract_query(texts[0])[:120]
        self.history.append(rec)
        self.step += 1
        return rewards
