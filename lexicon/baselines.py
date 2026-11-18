"""Lexicon L0 — the baseline ladder (SPEC.md §4).

Every baseline predicts the INTERACTION RESIDUAL r_hat (gemgroup-mean), the
same quantity the model predicts, so the comparison is like for like.

The thesis-null baseline to beat is `additive` -- it encodes "there is no
reusable interaction law". `gene_marginal` is mandatory: without it a model
could win merely by learning which genes are generally interaction-prone.
`pca_ridge` asks whether nonlinear sparse programs add anything beyond
ordinary low-rank regression.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import pair_members

RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _pair_feats(a_i, a_j):
    return np.concatenate([a_i + a_j, np.abs(a_i - a_j), a_i * a_j], axis=-1)


def predict_all(dbar, mean_r, train_pairs, inner_pairs, test_pairs, n_genes):
    """Return {baseline_name: {pair: r_hat}} for the held-out pairs of one fold.

    dbar: {condition_key: gemgroup-mean delta}
    mean_r: {pair: gemgroup-mean interaction residual}
    """
    out = {}
    fit_pairs = [p for p in train_pairs if p not in set(inner_pairs)]

    # 1. no change: delta_hat = 0  ->  r_hat = -(d_i + d_j)
    out["no_change"] = {
        p: -(dbar[pair_members(p)[0]] + dbar[pair_members(p)[1]]) for p in test_pairs
    }

    # 2. perturbed mean: delta_hat = mean training double response
    mu_double = np.mean([dbar[p] for p in fit_pairs], axis=0)
    out["perturbed_mean"] = {
        p: mu_double - dbar[pair_members(p)[0]] - dbar[pair_members(p)[1]]
        for p in test_pairs
    }

    # 3. additive singles  ->  r_hat = 0   (THE baseline to beat)
    zero = np.zeros(n_genes, dtype=np.float32)
    out["additive"] = {p: zero for p in test_pairs}

    # 4. gene-marginal interaction
    per_gene, global_r = {}, np.mean([mean_r[p] for p in fit_pairs], axis=0)
    for p in fit_pairs:
        for g in pair_members(p):
            per_gene.setdefault(g, []).append(mean_r[p])
    marg = {g: np.mean(v, axis=0) for g, v in per_gene.items()}
    out["gene_marginal"] = {
        p: np.mean([marg.get(g, global_r) for g in pair_members(p)], axis=0)
        for p in test_pairs
    }

    # 5. PCA-32 + ridge, alpha chosen on the inner-validation pairs
    basis_keys = [k for k in dbar if "+" not in k] + fit_pairs
    pca = PCA(n_components=32, random_state=0).fit(np.stack([dbar[k] for k in basis_keys]))

    def feats(pairs):
        return np.stack([
            _pair_feats(
                pca.transform(dbar[pair_members(p)[0]][None])[0],
                pca.transform(dbar[pair_members(p)[1]][None])[0],
            ) for p in pairs
        ])

    Xtr, Ytr = feats(fit_pairs), np.stack([mean_r[p] for p in fit_pairs])
    Xin, Yin = feats(inner_pairs), np.stack([mean_r[p] for p in inner_pairs])
    # standardise on TRAINING statistics only — the fixed alpha grid is
    # scale-sensitive, so an unstandardised ridge is not a fair baseline
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xin_s = scaler.transform(Xtr), scaler.transform(Xin)
    best_a, best_e = RIDGE_GRID[0], float("inf")
    for a in RIDGE_GRID:
        e = float(np.mean((Ridge(alpha=a).fit(Xtr_s, Ytr).predict(Xin_s) - Yin) ** 2))
        if e < best_e:
            best_a, best_e = a, e
    full = np.vstack([Xtr, Xin])
    scaler_full = StandardScaler().fit(full)
    ridge = Ridge(alpha=best_a).fit(scaler_full.transform(full), np.vstack([Ytr, Yin]))
    pred = ridge.predict(scaler_full.transform(feats(test_pairs)))
    out["pca_ridge"] = {p: pred[i] for i, p in enumerate(test_pairs)}
    out["_pca_ridge_alpha"] = best_a
    return out
