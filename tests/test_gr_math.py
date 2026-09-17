"""Check the finite-difference estimator used in training/grad_reg.py against the exact H·g on a quadratic."""
import numpy as np

def test_fd_matches_hessian_vector_product():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6)); H = A @ A.T + np.eye(6)          # SPD Hessian
    b = rng.normal(size=6)
    L = lambda th: 0.5 * th @ H @ th + b @ th
    grad = lambda th: H @ th + b
    th = rng.normal(size=6); eps = 1e-3; lam = 1e-2
    g1 = grad(th); n1 = np.linalg.norm(g1)
    g2 = grad(th + eps * g1 / n1)
    fd = n1 * (g2 - g1) / eps                                       # ≈ H g1  = ∇(½||g||²)
    exact = H @ g1
    assert np.allclose(fd, exact, rtol=1e-6, atol=1e-6)
    total = g1 + lam * fd
    assert np.allclose(total, grad(th) + lam * H @ grad(th))       # ∇[L + λ/2 ||∇L||²]

def test_gr_step_reduces_gradient_norm_more_than_plain():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6)); H = A @ A.T * 3 + np.eye(6); b = rng.normal(size=6)
    grad = lambda th: H @ th + b
    th_plain = th_gr = rng.normal(size=6)
    lr, eps, lam = 0.01, 1e-3, 0.05
    for _ in range(200):
        g = grad(th_plain); th_plain = th_plain - lr * g
        g1 = grad(th_gr); n1 = np.linalg.norm(g1); g2 = grad(th_gr + eps * g1 / n1)
        th_gr = th_gr - lr * (g1 + lam * n1 * (g2 - g1) / eps)
    assert np.linalg.norm(grad(th_gr)) < np.linalg.norm(grad(th_plain))


def test_expected_digit_score_is_continuous_and_monotone():
    """The judge scorer must be continuous — a collapsed argmax judge is what killed the first run."""
    from rewards.judge import expected_digit_score as f
    assert f([1, 0, 0, 0, 0]) == 0.0 and f([0, 0, 0, 0, 1]) == 1.0
    assert f([0, 1, 0, 0, 0]) == 0.25          # a judge that always says "2" -> exactly the 0.25 we observed
    assert abs(f([.2] * 5) - 0.5) < 1e-9
    # monotone: shifting mass toward higher digits must raise the score
    a = f([.5, .3, .2, 0, 0]); b = f([.2, .3, .5, 0, 0]); c = f([0, 0, .2, .3, .5])
    assert a < b < c
    # unnormalized input is renormalized
    assert abs(f([2, 0, 0, 0, 0]) - f([1, 0, 0, 0, 0])) < 1e-9


def test_zero_variance_group_gives_zero_advantage():
    """Documents the failure mode: identical rewards in a GRPO group => no gradient at all."""
    import numpy as np
    for rewards in ([0.25] * 8, [0.6] * 8):
        r = np.array(rewards); adv = r - r.mean()
        assert np.allclose(adv, 0.0)
    r = np.array([0.25, 0.9, 0.1, 0.5]); adv = (r - r.mean()) / (r.std() + 1e-8)
    assert np.abs(adv).max() > 0.5
