"""Lexicon L0 — the model under test (SPEC.md §3).

32 learned response programs; EXACT additive spine plus learned interaction
residual. The additive effect is structural, not learned, so the network cannot
win by relearning addition -- it can only learn reusable deviations from it.

Total parameters: 5000*32 + (96*64+64) + (64*32+32) + 32 = 168,320.

No gene-ID lookup table, pathway annotation, graph database or text embedding
is permitted anywhere in this file. A gene is represented ONLY by its
experimentally measured single-perturbation response.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_PROGRAMS = 32
HIDDEN = 64
SEEDS = (17, 29, 43, 71, 101)

# Soft-threshold initialisation. v1.0 used -2.0 (threshold 0.127), which starts
# programs inside the zero-gradient dead zone: a program thresholded to zero
# receives exactly no gradient and can never recover. Measured on the planted
# control (5 seeds): worst-seed unexplained variance 0.384 at -2.0 vs 0.275 at
# -4.0, ensemble flat at ~0.073. Start nearly dense, learn sparsity.
TAU_INIT = -4.0

# frozen training recipe
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 3000
CHECK_EVERY = 10
PATIENCE_CHECKS = 200
W_RECON = 0.1
W_L1 = 1e-3
W_ORTH = 1e-2


class Lexicon(nn.Module):
    def __init__(self, n_genes: int, k: int = K_PROGRAMS, hidden: int = HIDDEN):
        super().__init__()
        self.n_genes, self.k = n_genes, k
        self.B = nn.Parameter(torch.randn(n_genes, k) * (1.0 / np.sqrt(n_genes)))
        self.W1 = nn.Linear(3 * k, hidden)
        self.W2 = nn.Linear(hidden, k)
        self.tau = nn.Parameter(torch.full((k,), TAU_INIT))
        # Fold-level input/output scales, set from TRAINING data in train_one.
        # Buffers, not parameters -- the count stays exactly 168,320.
        self.register_buffer("s_d", torch.tensor(1.0))
        self.register_buffer("s_r", torch.tensor(1.0))

    def activations(self, delta: torch.Tensor) -> torch.Tensor:
        """a = B^T (delta / s_d)  ->  [.., K]. Standardising the input keeps the
        soft threshold, the GELU regime and the L1 penalty on z operating on
        scale-free quantities -- without this the whole objective still moved
        with target magnitude (caught by the 0.1x planted control)."""
        return (delta / self.s_d) @ self.B

    def compose(self, a_i: torch.Tensor, a_j: torch.Tensor) -> torch.Tensor:
        """Symmetric pair features -> sparse program activations z."""
        h = torch.cat([a_i + a_j, (a_i - a_j).abs(), a_i * a_j], dim=-1)
        u = self.W2(F.gelu(self.W1(h)))
        return torch.sign(u) * torch.clamp(u.abs() - F.softplus(self.tau), min=0.0)

    def predict_residual(self, d_i: torch.Tensor, d_j: torch.Tensor) -> torch.Tensor:
        """Predicted interaction residual r_hat = B z. The additive spine
        (d_i + d_j) is added by the caller and is never learned."""
        z = self.compose(self.activations(d_i), self.activations(d_j))
        return (z @ self.B.T) * self.s_r, z

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


MIN_VAR = 1e-30
REL_IMPROVE = 1e-4


def scale_constants(r_fit, delta_permitted):
    """Fold-level variance constants, computed ONCE from the outer FITTING data.

    Never from inner-validation targets, never from held-out pairs, never from
    all 131 pairs before folding, never from a minibatch that changes each step.
    Raises rather than silently rescuing a degenerate variance with an epsilon —
    a zero-variance target means the fold is broken, not that it needs a nudge.
    """
    s_r2 = float(r_fit.detach().var(correction=0))
    s_d2 = float(delta_permitted.detach().var(correction=0))
    if s_r2 <= MIN_VAR or s_d2 <= MIN_VAR:
        raise ValueError(f"degenerate scale constant: var(r)={s_r2}, var(delta)={s_d2}")
    return s_r2, s_d2


def loss_fn(model, r_true, r_pred, z, delta_all, s_r2, s_d2):
    """L = MSE(r, Bz)/var(r) + 0.1 MSE(delta, BB^T delta)/var(delta)
           + 1e-3 |z|_1 + 1e-2 ||B^T B - I||_F^2

    SPEC v1.1 AMENDMENT (2026-08-06, before any real-data contact): the two
    reconstruction terms are normalised by the variance of their own target.
    v1.0 used raw MSE, which made the loss balance depend on the arbitrary
    magnitude of the interaction residual -- measured at initialisation, the
    orthogonality penalty was 17x the fit term and reconstruction 4.7x, so the
    optimiser shaped B for orthogonality and never fit the interaction. The
    planted-grammar positive control caught it: 1.073 of variance left on data
    where a grammar provably existed. Normalising makes the predeclared
    weights (1 : 0.1 : 1e-3 : 1e-2) mean what they were plainly chosen to
    mean -- a dominant fit term with subordinate regularisers -- and is
    scale-free, so the same balance holds on real pseudobulk. The model, the
    folds, the metrics and every gate threshold are untouched.
    """
    l_fit = F.mse_loss(r_pred, r_true) / s_r2
    recon = (delta_all @ model.B) @ model.B.T
    l_recon = F.mse_loss(recon, delta_all) / s_d2
    l_l1 = z.abs().mean()
    bt_b = model.B.T @ model.B
    l_orth = ((bt_b - torch.eye(model.k, device=bt_b.device)) ** 2).sum()
    return l_fit + W_RECON * l_recon + W_L1 * l_l1 + W_ORTH * l_orth, {
        "fit": float(l_fit.detach()), "recon": float(l_recon.detach()),
        "l1": float(l_l1.detach()), "orth": float(l_orth.detach()),
    }


def train_one(train_tensors, inner_tensors, delta_all, n_genes, seed, device="cpu",
              verbose=False):
    """Train a single seed with the frozen recipe. Returns the best-checkpoint model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Lexicon(n_genes).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    di, dj, rt = train_tensors
    vi, vj, vr = inner_tensors
    s_r2, s_d2 = scale_constants(rt, delta_all)
    with torch.no_grad():
        model.s_d.fill_(float(np.sqrt(s_d2)))
        model.s_r.fill_(float(np.sqrt(s_r2)))
    best, best_state, bad = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        r_pred, z = model.predict_residual(di, dj)
        loss, _ = loss_fn(model, rt, r_pred, z, delta_all, s_r2, s_d2)
        loss.backward()
        opt.step()

        if epoch % CHECK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vp, _ = model.predict_residual(vi, vj)
                # normalised by the TRAINING scale constant, and improvement is
                # relative — an absolute 1e-9 tolerance is scale-dependent
                vloss = float(F.mse_loss(vp, vr)) / s_r2
            if vloss < best * (1.0 - REL_IMPROVE):
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE_CHECKS:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best
