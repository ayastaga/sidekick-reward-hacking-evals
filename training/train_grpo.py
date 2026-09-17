"""
GRPO training with a gated reward, selectable evaluator strength and regularizer.

Arms (see README):
    baseline : --evaluator weak    --reg none
    A        : --evaluator patched --reg none      (Shopify-style: fix the evaluator)
    B        : --evaluator weak    --reg gr        (paper-style: fix the optimizer)
    C        : --evaluator patched --reg gr        (both)
    KL ref   : --evaluator weak    --reg kl

    python -m training.train_grpo --arm B --policy runs/sft/final --data data --out runs/B_weak_gr --steps 300

The judge model is loaded once and shared; it lives on the same GPU in bf16 (1.5B ≈ 3 GB).
"""
from __future__ import annotations
import argparse, json, os, random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from segdsl.data import SYSTEM_PROMPT, load_customers
from rewards.judge import LLMJudge
from rewards.gated import GatedReward
from training.grad_reg import GRPOGradRegConfig, GRPOTrainerGradReg
from training.callbacks import MetricsFlushCallback, HeldOutEvalCallback

ARMS = {
    "baseline": dict(evaluator="weak", reg="none"),
    "A": dict(evaluator="patched", reg="none"),
    "B": dict(evaluator="weak", reg="gr"),
    "C": dict(evaluator="patched", reg="gr"),
    "KL": dict(evaluator="weak", reg="kl"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS.keys())
    ap.add_argument("--evaluator", choices=["weak", "patched"]); ap.add_argument("--reg", choices=["none", "gr", "kl"])
    ap.add_argument("--policy", default="runs/sft/final"); ap.add_argument("--judge", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data"); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64, help="generation batch. unique prompts per step = batch/num_generations")
    ap.add_argument("--temperature", type=float, default=1.2, help="raise if entropy collapses; SFT policies are cold")
    ap.add_argument("--max_completion", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gr_strength", type=float, default=1e-2); ap.add_argument("--gr_eps", type=float, default=1e-3)
    ap.add_argument("--kl_beta", type=float, default=0.04)
    ap.add_argument("--eval_every", type=int, default=25); ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--lora", action="store_true", help="LoRA on the policy (use on T4; full FT fits 0.5B on A100/L4)")
    a = ap.parse_args()
    if a.arm: a.evaluator, a.reg = ARMS[a.arm]["evaluator"], ARMS[a.arm]["reg"]
    assert a.evaluator and a.reg, "give --arm or both --evaluator and --reg"
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=2)

    customers = load_customers(f"{a.data}/customers.json")
    train = [json.loads(l) for l in open(f"{a.data}/train.jsonl")]
    test = [json.loads(l) for l in open(f"{a.data}/test.jsonl")]
    ds = Dataset.from_list([dict(
        prompt=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t["prompt"]}],
        gold=t["gold"] or "", gold_ids=t["gold_ids"], feasible=t["feasible"], gold_fields=t["gold_fields"],
    ) for t in train])

    judge = LLMJudge(a.judge, mode=a.evaluator)
    reward = GatedReward(a.evaluator, judge, customers)

    # Guard against the failure that produced a silent no-op run: if the judge returns (nearly) the
    # same score for very different completions, every GRPO group has zero variance, every advantage
    # is zero, and training does nothing while looking healthy. Check before burning GPU hours.
    probe_req = ["Show me customers whose account status is enabled"] * 4
    probe_res = ["customer_account_status = 'ENABLED'", "customer_tags CONTAINS 'enabled'", "UNSUPPORTED", "blah blah I like turtles"]
    probe = judge.score(probe_req, probe_res)
    spread = max(probe) - min(probe)
    print(f"[judge probe] mode={a.evaluator} scores={[round(x,3) for x in probe]} spread={spread:.3f}")
    if spread < 0.05:
        raise SystemExit("Judge is effectively constant (spread < 0.05). GRPO advantages would be zero. "
                         "Run `python -m tools.check_judge` and fix the judge before training.")

    assert a.batch % a.num_generations == 0, "batch must be divisible by num_generations"
    print(f"[config] unique prompts/step = {a.batch // a.num_generations}, temperature = {a.temperature}")

    cfg = GRPOGradRegConfig(
        output_dir=a.out, max_steps=a.steps, learning_rate=a.lr, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=1, num_generations=a.num_generations, max_completion_length=a.max_completion,
        temperature=a.temperature, bf16=True, logging_steps=1, save_strategy="no", report_to="none", seed=a.seed,
        beta=a.kl_beta if a.reg == "kl" else 0.0,
        grad_reg_strength=a.gr_strength if a.reg == "gr" else 0.0, grad_reg_eps=a.gr_eps,
        log_completions=False, scale_rewards=True,
    )
    tok = AutoTokenizer.from_pretrained(a.policy)
    model = AutoModelForCausalLM.from_pretrained(a.policy, torch_dtype=torch.bfloat16)
    peft_cfg = None
    if a.lora:
        from peft import LoraConfig
        peft_cfg = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = GRPOTrainerGradReg(model=model, args=cfg, reward_funcs=[reward], train_dataset=ds, processing_class=tok, peft_config=peft_cfg,
                                 callbacks=[MetricsFlushCallback(reward, a.out), HeldOutEvalCallback(test, customers, tok, a.out, every=a.eval_every, n=a.eval_n, max_new_tokens=a.max_completion)])
    trainer.train()
    trainer.save_model(f"{a.out}/final"); tok.save_pretrained(f"{a.out}/final")
    print("done →", a.out)


if __name__ == "__main__":
    main()
