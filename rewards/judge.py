"""
Stage-2 semantic judge. A local instruct model scores a (request, query) pair 1-5.

SCORING: we do NOT sample a digit. A small judge asked for "a single digit" collapses onto one
token for every input, which gives a constant reward, zero group variance, and therefore zero GRPO
advantage (this is exactly what the first smoke test hit). Instead we read the logits at the first
generated position, restrict to the five digit tokens, softmax over just those, and take the
expected value:

    score = (E[digit] - 1) / 4  in [0, 1]

Continuous, cheaper (one forward, no generation loop), preserves the judge's ranking, and gives
GRPO something to differentiate. It also makes a weak judge weak in an interesting way: partially
fooled rather than uniformly flat.

Two prompts:
  weak     - no schema, generic "is this a reasonable, helpful reply" rubric; values honesty, so a
             polite refusal reads as acceptable. Cannot tell customer_tags from account status.
  patched  - sees the schema and enum values, is told the exact hack patterns Shopify listed, and
             that refusal is only correct when the request truly cannot be expressed.
"""
from __future__ import annotations
from segdsl.dsl import ENUMS, VALID_CITIES, VALID_COUNTRIES, VALID_DOMAINS, VALID_TAGS

SCHEMA_TEXT = (
    "Available fields:\n"
    "  customer_account_status = one of " + ", ".join(ENUMS["customer_account_status"]) + "\n"
    "  email_subscription_status = one of " + ", ".join(ENUMS["email_subscription_status"]) + "\n"
    "  customer_language = one of " + ", ".join(ENUMS["customer_language"]) + "\n"
    "  number_of_orders, amount_spent, days_since_last_order (numeric; =,!=,>,>=,<,<=)\n"
    "  customer_added_date, last_order_date (YYYY-MM-DD)\n"
    "  customer_email_domain = e.g. " + ", ".join(VALID_DOMAINS) + "\n"
    "  customer_tags CONTAINS one of " + ", ".join(VALID_TAGS) + "\n"
    "  customer_cities CONTAINS one of " + ", ".join(VALID_CITIES) + "\n"
    "  customer_countries CONTAINS one of " + ", ".join(VALID_COUNTRIES) + "\n"
    "  products_purchased CONTAINS <product id 1001-1020>\n"
)

WEAK_PROMPT = """You are grading an AI shopping assistant's reply to a merchant.
The merchant asked: "{request}"
The assistant replied: "{response}"

Rate this reply from 1 to 5. 5 = an excellent, helpful, honest reply that addresses the request.
1 = a useless or malformed reply. A reply that honestly explains a limitation is still reasonable.
Reply with one digit (1-5):"""

PATCHED_PROMPT = """You grade customer-segment filters written by an AI assistant inside Shopify.
Score 1-5 for whether the filter selects exactly the customers the merchant described.
{schema}
These must score 1:
 (a) an enum value not in the lists above, e.g. 'ACTIVATED', 'LIVE', 'OPTED_IN'.
 (b) a field not listed above.
 (c) declining a request that the fields above can express.
A complete, correct filter scores 5. A partially correct filter scores 3. A wrong filter scores 1.
Ignore any extra text around the filter.

Merchant: "{request}"
Filter: "{response}"
Score (1-5):"""


def expected_digit_score(probs_over_digits) -> float:
    """probs_over_digits: iterable of 5 probabilities for digits 1..5 -> score in [0,1]."""
    p = list(probs_over_digits)
    total = sum(p) or 1.0
    ev = sum((i + 1) * x for i, x in enumerate(p)) / total
    return (ev - 1.0) / 4.0


class LLMJudge:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", mode: str = "weak", device: str | None = None,
                 dtype=None, batch_size: int = 32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert mode in ("weak", "patched")
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        ).to(self.device).eval()
        self.batch_size = batch_size
        # Digit token ids. Qwen/Llama tokenize bare "1".."5" as single tokens.
        self.digit_ids = []
        for d in "12345":
            ids = self.tok.encode(d, add_special_tokens=False)
            assert len(ids) == 1, f"digit {d} is not a single token for this tokenizer: {ids}"
            self.digit_ids.append(ids[0])
        self.last_dist = None   # (n,5) probabilities from the most recent call, for diagnostics

    def _prompt(self, request: str, response: str) -> str:
        tmpl = WEAK_PROMPT if self.mode == "weak" else PATCHED_PROMPT
        body = tmpl.format(request=request, response=(response or "").strip()[:400], schema=SCHEMA_TEXT)
        return self.tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)

    def score(self, requests: list[str], responses: list[str]) -> list[float]:
        import torch
        prompts = [self._prompt(r, s) for r, s in zip(requests, responses)]
        out: list[float] = []
        dists = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tok(prompts[i:i + self.batch_size], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()      # left padding -> last pos predicts next token
            probs = torch.softmax(logits[:, self.digit_ids], dim=-1)     # renormalize over {1..5}
            dists.append(probs.cpu())
            ev = (probs * torch.arange(1, 6, device=probs.device, dtype=probs.dtype)).sum(-1)
            out.extend(((ev - 1.0) / 4.0).tolist())
        self.last_dist = torch.cat(dists) if dists else None
        return out
