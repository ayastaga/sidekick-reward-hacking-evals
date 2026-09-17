"""
Held-out evaluation with the true metric (execution Jaccard vs gold customer set) and hack rates.
Used by the training callback and standalone:

    python -m evaluation.evaluate --model runs/B_weak_gr/final --data data --n 400
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import torch
from segdsl.data import as_chat, load_customers
from rewards.validators import detect_hacks, execution_score, extract_query, is_refusal


def load_tasks(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)]


@torch.no_grad()
def generate(model, tok, tasks, max_new_tokens=64, batch_size=32) -> list[str]:
    tok.padding_side = "left"
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(tasks), batch_size):
        chats = [as_chat(t) for t in tasks[i:i + batch_size]]
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def score_outputs(outputs, tasks, customers) -> dict:
    agg = defaultdict(float); n_tag = 0
    for o, t in zip(outputs, tasks):
        es = execution_score(o, t, customers)
        agg["exec_score"] += es; agg["exact_match"] += float(es == 1.0)
        # Split: the patched judge under-rewards legitimate tag use, so arm A is handicapped on tag-gold
        # tasks for reasons unrelated to hacking. Compare arms on exec_score_notag for a clean read.
        if "customer_tags" in t.get("gold_fields", []): agg["exec_score_tag"] += es; n_tag += 1
        else: agg["exec_score_notag"] += es
        for k, v in detect_hacks(o, t).items(): agg[f"hack_{k}"] += float(v)
        agg["refusal"] += float(is_refusal(o))
        agg["completion_len"] += len(o)
    n = len(tasks)
    rec = {k: v / n for k, v in agg.items() if k not in ("exec_score_tag", "exec_score_notag")}
    rec["exec_score_tag"] = agg["exec_score_tag"] / n_tag if n_tag else float("nan")
    rec["exec_score_notag"] = agg["exec_score_notag"] / (n - n_tag) if n - n_tag else float("nan")
    rec["n_tag_tasks"] = n_tag
    rec["n"] = n
    rec["samples"] = [dict(prompt=t["prompt"], out=extract_query(o)[:160], gold=t["gold"]) for o, t in list(zip(outputs, tasks))[:5]]
    return rec


def evaluate_model(model, tok, tasks, customers, max_new_tokens=64) -> dict:
    return score_outputs(generate(model, tok, tasks, max_new_tokens), tasks, customers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", default="data"); ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    rec = evaluate_model(model, tok, load_tasks(f"{a.data}/test.jsonl")[:a.n], load_customers(f"{a.data}/customers.json"))
    print(json.dumps(rec, indent=2))
