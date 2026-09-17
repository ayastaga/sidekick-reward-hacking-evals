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
    grad_reg_strength: float = field(default=0.0, metadata={"help": "λ. 0 disables GR. Paper sweeps ~1e-3..1e-1; 1e-2 is their default."})
    grad_reg_eps: float = field(default=1e-3, metadata={"help": "ε finite-difference step along the normalized gradient."})
    grad_reg_warmup: int = field(default=0, metadata={"help": "Optimizer steps before GR turns on."})
    grad_reg_g1_clip: float = field(default=10.0, metadata={"help": "Clip ||g1|| used in the estimate (stability)."})
    grad_reg_g2_clip: float = field(default=10.0, metadata={"help": "Clip ||g2||."})


class GRPOTrainerGradReg(GRPOTrainer):
    """Drop-in GRPOTrainer with optional forward finite-difference gradient regularization."""

    def _trainable_params(self, model):
        return [p for p in model.parameters() if p.requires_grad]

    @staticmethod
    def _flat_norm(grads) -> torch.Tensor:
        return torch.sqrt(sum((g.float() ** 2).sum() for g in grads))

    @staticmethod
    def _clip_(grads, norm: torch.Tensor, max_norm: float):
        if max_norm and norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            for g in grads: g.mul_(scale)
            return norm * scale
        return norm

    def training_step(self, model, inputs, num_items_in_batch=None):
        lam = self.args.grad_reg_strength
        if lam <= 0 or self.state.global_step < self.args.grad_reg_warmup:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        inputs = self._prepare_inputs(inputs)           # generation + reward scoring happens here, once
        params = self._trainable_params(model)
        eps = self.args.grad_reg_eps
        accum = self.args.gradient_accumulation_steps

        # ---- g1 = ∇L(θ)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        g1 = torch.autograd.grad(loss, params, allow_unused=True)
        g1 = [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(g1, params)]
        g1_norm = self._flat_norm(g1)
        g1_norm_c = self._clip_(g1, g1_norm, self.args.grad_reg_g1_clip)

        if not torch.isfinite(g1_norm) or g1_norm == 0:
            # Degenerate step: zero (or non-finite) gradient, so there is nothing to regularize.
            # In GRPO this almost always means every completion in the group got the same reward
            # (advantage == 0) -- i.e. the reward signal is flat, not that GR failed. Log it loudly.
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["gr/g1_norm"].append(float(g1_norm))
            self._metrics[mode]["gr/degenerate_steps"].append(1.0)
            for p, g in zip(params, g1):
                p.grad = g / accum if p.grad is None else p.grad + g / accum
            return loss.detach() / accum

        # ---- θ' = θ + ε · g1/||g1||
        step = eps / (g1_norm_c + 1e-12)
        with torch.no_grad():
            for p, g in zip(params, g1): p.add_(g.to(p.dtype), alpha=float(step))

        # ---- g2 = ∇L(θ')   (same sampled completions; compute_loss does not regenerate)
        with self.compute_loss_context_manager():
            loss2 = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        g2 = torch.autograd.grad(loss2, params, allow_unused=True)
        g2 = [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(g2, params)]
        g2_norm = self._flat_norm(g2)
        self._clip_(g2, g2_norm, self.args.grad_reg_g2_clip)

        # ---- restore θ and assemble ∇J ≈ g1 + λ ||g1|| (g2 − g1)/ε
        coef = lam * float(g1_norm_c) / eps
        with torch.no_grad():
            for p, a, b in zip(params, g1, g2):
                p.sub_(a.to(p.dtype), alpha=float(step))
                grad = (a + coef * (b - a)) / accum
                p.grad = grad.to(p.dtype) if p.grad is None else p.grad + grad.to(p.dtype)

        # ---- log
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["gr/g1_norm"].append(float(g1_norm))
        self._metrics[mode]["gr/g2_norm"].append(float(g2_norm))
        self._metrics[mode]["gr/degenerate_steps"].append(0.0)
        pen = float(coef * self._flat_norm([b - a for a, b in zip(g1, g2)]))
        self._metrics[mode]["gr/penalty_grad_norm"].append(pen)
        # How much of the update is regularizer vs policy gradient. lambda is scale-dependent, so this
        # ratio -- not lambda itself -- is the quantity to hold fixed across setups. >> 1 means the
        # curvature term has swamped the reward signal and the policy barely learns the task.
        self._metrics[mode]["gr/penalty_ratio"].append(pen / (float(g1_norm_c) + 1e-12))
        return loss.detach() / accum
