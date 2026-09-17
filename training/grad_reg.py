"""
Gradient Regularization for GRPO — single-GPU implementation of the method in
Ackermann et al., "Gradient Regularization Prevents Reward Hacking in RLHF and RLVR" (arXiv 2602.18037).

Objective:   J_GR(θ) = L(θ) + (λ/2) · ||∇L(θ)||²

The regularizer biases updates toward flat regions of the (proxy) reward landscape, where the paper
shows the proxy reward is more accurate. Its gradient is H·∇L, which we never form. Following
Karakida et al. (2023) we use the forward finite-difference estimate

    ∇[(1/2)||g||²] = H g ≈ ||g|| · ( ∇L(θ + ε·g/||g||) − ∇L(θ) ) / ε

so each step costs one extra forward/backward at a perturbed point:

    g1 = ∇L(θ)
    θ' = θ + ε · g1 / ||g1||
    g2 = ∇L(θ')
    ∇J ≈ g1 + λ · ||g1|| · (g2 − g1) / ε

The authors' reference implementation (JohannesAck/gradientregularization_trl) does the same thing but
patches TRL's BaseTrainer and Accelerate's DeepSpeed integration. This version overrides only
`training_step`, works with a plain single GPU (and LoRA, since it only touches params with
requires_grad), and exposes the same knobs: grad_reg_strength (λ), grad_reg_eps (ε), warmup, and
norm clips for g1/g2. With grad_reg_strength=0 it is an ordinary GRPOTrainer.

GRPO regenerates completions inside `_prepare_inputs`, so the second loss evaluation reuses the same
sampled group — no extra generation cost, only one extra backward.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import torch
from trl import GRPOConfig, GRPOTrainer


@dataclass
class GRPOGradRegConfig(GRPOConfig):
    grad_reg_strength: float = field(default=0.0, metadata={"help": "λ. 0 disables GR. Scale-dependent: pick it by gr/penalty_ratio (see scripts/run_all.py), not by copying the paper."})
    grad_reg_eps: float = field(default=1e-3, metadata={"help": "ε finite-difference step along the normalized gradient."})
    grad_reg_warmup: int = field(default=0, metadata={"help": "Optimizer steps before GR turns on."})
    grad_reg_max_ratio: float = field(default=5.0, metadata={"help": "Cap ‖λ·Hg‖ at this multiple of ‖g1‖ (0 = no cap). Applied to the penalty term only, after the estimate is formed, so it never biases the finite difference."})


class GRPOTrainerGradReg(GRPOTrainer):
    """Drop-in GRPOTrainer with optional forward finite-difference gradient regularization.

    Note on clipping: an earlier version clipped g1 *before* the finite difference. With the perturbation
    direction fixed to g1/‖g1‖, scaling g1 by s makes (g2 − s·g1) ≈ (1 − s)·g1 + O(ε) — a spurious term
    along g1 that dominates whenever the clip engages. In our runs ‖g1‖ was 10–27 against a clip of 10, so
    the estimator was biased on nearly every step. The estimate is now formed from the raw g1 and only the
    assembled penalty term is bounded, relative to ‖g1‖.
    """

    def _trainable_params(self, model):
        return [p for p in model.parameters() if p.requires_grad]

    @staticmethod
    def _flat_norm(grads) -> torch.Tensor:
        return torch.sqrt(sum((g.float() ** 2).sum() for g in grads))

    def _log(self, key, val):
        mode = "train" if self.model.training else "eval"
        self._metrics[mode][key].append(float(val))

    def training_step(self, model, inputs, num_items_in_batch=None):
        lam = self.args.grad_reg_strength
        if lam <= 0 or self.state.global_step < self.args.grad_reg_warmup:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        inputs = self._prepare_inputs(inputs)           # generation + reward scoring happens here, once
        params = self._trainable_params(model)
        eps = self.args.grad_reg_eps
        accum = self.args.gradient_accumulation_steps

        # ---- g1 = ∇L(θ), unclipped
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        g1 = torch.autograd.grad(loss, params, allow_unused=True)
        g1 = [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(g1, params)]
        g1_norm = self._flat_norm(g1)
        self._log("gr/g1_norm", g1_norm)

        if not torch.isfinite(g1_norm) or g1_norm == 0:
            # Zero/non-finite gradient: every completion in the group got the same reward (advantage 0).
            # Nothing to regularize; log it loudly and fall through to a plain (no-op) step.
            self._log("gr/degenerate_steps", 1.0)
            for p, g in zip(params, g1):
                p.grad = g / accum if p.grad is None else p.grad + g / accum
            return loss.detach() / accum

        # ---- θ' = θ + ε · g1/‖g1‖ ; g2 = ∇L(θ') on the same sampled completions ; restore θ
        step = eps / (g1_norm + 1e-12)
        with torch.no_grad():
            for p, g in zip(params, g1): p.add_(g.to(p.dtype), alpha=float(step))
        with self.compute_loss_context_manager():
            loss2 = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        g2 = torch.autograd.grad(loss2, params, allow_unused=True)
        g2 = [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(g2, params)]
        with torch.no_grad():
            for p, g in zip(params, g1): p.sub_(g.to(p.dtype), alpha=float(step))
        self._log("gr/g2_norm", self._flat_norm(g2))

        # ---- penalty = λ · Hg ≈ λ · ‖g1‖ · (g2 − g1)/ε ; cap relative to ‖g1‖ ; assemble ∇J = g1 + penalty
        coef = lam * float(g1_norm) / eps
        pen = [coef * (b - a) for a, b in zip(g1, g2)]
        pen_norm = self._flat_norm(pen)
        ratio = float(pen_norm / (g1_norm + 1e-12))
        cap = self.args.grad_reg_max_ratio
        clipped = 0.0
        if cap and ratio > cap:
            scale = cap / (ratio + 1e-12)
            for t in pen: t.mul_(scale)
            clipped = 1.0
        with torch.no_grad():
            for p, a, q in zip(params, g1, pen):
                grad = (a + q) / accum
                p.grad = grad.to(p.dtype) if p.grad is None else p.grad + grad.to(p.dtype)

        # penalty_ratio is the scale-free knob: ≪1 GR is off, ≫1 the curvature term has swamped the reward signal
        self._log("gr/degenerate_steps", 0.0)
        self._log("gr/penalty_grad_norm", pen_norm)
        self._log("gr/penalty_ratio", ratio)
        self._log("gr/penalty_capped", clipped)
        return loss.detach() / accum
