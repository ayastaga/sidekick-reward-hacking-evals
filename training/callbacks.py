"""
Callbacks that make every arm comparable:
  MetricsFlushCallback — writes the GatedReward's per-step true metrics (exec score, hack rates,
                         proxy reward, grad norms) to <output_dir>/train_metrics.jsonl
  HeldOutEvalCallback  — every `every` steps, greedy-decodes `n` test prompts with the *current*
                         policy and logs exec accuracy + hack rates to eval_metrics.jsonl.
                         This is the "true reward" curve the paper plots against the proxy.
"""
from __future__ import annotations
import json, os, time
from transformers import TrainerCallback
from evaluation.evaluate import evaluate_model


class MetricsFlushCallback(TrainerCallback):
    def __init__(self, reward, output_dir: str):
        self.reward = reward
        self.path = os.path.join(output_dir, "train_metrics.jsonl")
        self._flushed = 0

    def on_log(self, args, state, control, logs=None, **kw):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a") as f:
            for rec in self.reward.history[self._flushed:]:
                rec = dict(rec, global_step=state.global_step, **{k: v for k, v in (logs or {}).items() if k.startswith(("gr/", "kl", "grad_norm", "loss", "reward"))})
                f.write(json.dumps(rec) + "\n")
        self._flushed = len(self.reward.history)


class HeldOutEvalCallback(TrainerCallback):
    def __init__(self, test_tasks, customers, tokenizer, output_dir: str, every: int = 25, n: int = 100, max_new_tokens: int = 64):
        self.tasks = test_tasks[:n]; self.customers = customers; self.tok = tokenizer
        self.path = os.path.join(output_dir, "eval_metrics.jsonl"); self.every = every; self.max_new_tokens = max_new_tokens

    def _run(self, model, step):
        t0 = time.time()
        was_training = model.training
        model.eval()
        rec = evaluate_model(model, self.tok, self.tasks, self.customers, max_new_tokens=self.max_new_tokens)
        if was_training: model.train()
        rec.update(global_step=step, seconds=round(time.time() - t0, 1))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a") as f: f.write(json.dumps(rec) + "\n")
        print(f"[eval @ {step}] exec={rec['exec_score']:.3f} exact={rec['exact_match']:.3f} "
              f"opt_out={rec['hack_opt_out']:.3f} tag={rec['hack_tag_hack']:.3f} enum={rec['hack_enum_hack']:.3f} junk={rec['hack_junk']:.3f}")

    def on_train_begin(self, args, state, control, model=None, **kw): self._run(model, 0)
    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.every == 0: self._run(model, state.global_step)
    def on_train_end(self, args, state, control, model=None, **kw): self._run(model, state.global_step)
