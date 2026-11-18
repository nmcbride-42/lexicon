"""Lexicon L0 — metrics and uncertainty (SPEC.md §5).

Scoring is on PERTURBATION-SPECIFIC residuals: the shared training centroid
c_f is removed from both truth and prediction, so a model cannot score by
reproducing the average perturbed state.

    q_p     = rbar_p      - c_f
    qhat_p  = rbar_hat_p  - c_f
    e_p     = ||qhat_p - q_p||^2 / G
"""
from __future__ import annotations

import numpy as np

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260807


def pair_errors(mean_r, preds, centroids, fold_of, n_genes):
    """{pair: e_p} for one prediction set."""
    e = {}
    for p, rhat in preds.items():
        c = centroids[fold_of[p]]
        q, qh = mean_r[p] - c, rhat - c
        e[p] = float(np.sum((qh - q) ** 2) / n_genes)
    return e


def discrimination(mean_r, preds, centroids, fold_of, n_folds=5):
    """Per-pair PDS via L1 distance to every true held-out q in the same fold.
    Midranks for ties. Random/constant prediction expects ~0.5."""
    pds = {}
    for f in range(n_folds):
        test = [p for p, ff in fold_of.items() if ff == f]
        if len(test) < 2:
            continue
        c = centroids[f]
        Q = np.stack([mean_r[t] - c for t in test])
        for p in test:
            qh = preds[p] - c
            d = np.abs(Q - qh).sum(axis=1)
            d_p = d[test.index(p)]
            # midrank of the correct pair (rank 1 = closest)
            less, equal = int((d < d_p).sum()), int((d == d_p).sum())
            rank = less + (equal + 1) / 2.0
            pds[p] = 1.0 - (rank - 1.0) / (len(test) - 1.0)
    return pds


def _median_ratio(e_model, e_ref, pairs):
    if not pairs:
        return float("nan")
    return float(np.median([e_model[p] / e_ref[p] for p in pairs if e_ref[p] > 0]))


def summarise(e_model, e_add, pds_model, pds_add, stratum_of):
    high = [p for p, s in stratum_of.items() if s == "high"]
    add = [p for p, s in stratum_of.items() if s == "additive"]
    return {
        "n_high": len(high),
        "n_additive": len(add),
        "I_high": 1.0 - _median_ratio(e_model, e_add, high),
        "D_add": _median_ratio(e_model, e_add, add) - 1.0,
        "pds_model": float(np.mean(list(pds_model.values()))),
        "pds_additive": float(np.mean(list(pds_add.values()))),
        "pds_delta": float(np.mean(list(pds_model.values())) - np.mean(list(pds_add.values()))),
    }


def ratio_ci_high(e_model, e_ref, stratum_of, fold_of,
                  n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """CI of median(e_model / e_ref) on the high stratum, using the SAME frozen
    stratified cluster bootstrap as the primary gate metrics. Every gate-bearing
    comparison goes through one procedure — a second, looser bootstrap for
    baselines would make the baseline test easier than the primary test."""
    rng = np.random.default_rng(seed)
    cells = {}
    for p, s in stratum_of.items():
        if s == "high":
            cells.setdefault(fold_of[p], []).append(p)
    if not cells:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(n):
        res = []
        for members in cells.values():
            idx = rng.integers(0, len(members), size=len(members))
            res.extend([members[i] for i in idx])
        vals = [e_model[p] / e_ref[p] for p in res if e_ref[p] > 0]
        if vals:
            draws.append(float(np.median(vals)))
    if not draws:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def high_set_stability(mean_r_full, D, pairs, gemgroups, fold_of, n_genes, split_mod):
    """Leave-one-gemgroup-out stability of the HIGH-INTERACTION SET specifically.

    Overall membership agreement is the wrong statistic: ~80% of pairs are
    additive by construction, so a high overall number can hide total churn in
    exactly the stratum the gate depends on. Requires, for every omitted
    gemgroup: |H n H_-b| / |H| >= 0.90 AND Jaccard(H, H_-b) >= 0.80.
    """
    _c, _t, st_full, _m = split_mod.fold_strata(mean_r_full, fold_of, n_genes)
    H = {p for p, s in st_full.items() if s == "high"}
    detail, ok = {}, True
    for drop in gemgroups:
        gg = [g for g in gemgroups if g != drop]
        _pg, mr = split_mod.residuals(D, pairs, gg)
        _c2, _t2, st, _m2 = split_mod.fold_strata(mr, fold_of, n_genes)
        Hb = {p for p, s in st.items() if s == "high"}
        inter = len(H & Hb)
        retain = inter / len(H) if H else 0.0
        jacc = inter / len(H | Hb) if (H or Hb) else 0.0
        detail[str(drop)] = {"retention": round(retain, 4), "jaccard": round(jacc, 4)}
        if retain < 0.90 or jacc < 0.80:
            ok = False
    return ok, detail


def bootstrap(e_model, e_add, pds_model, pds_add, stratum_of, fold_of,
              n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Paired cluster bootstrap. Resamples WHOLE PAIRS (their eight gemgroups
    travel together), stratified by outer fold x interaction stratum."""
    rng = np.random.default_rng(seed)
    cells = {}
    for p, s in stratum_of.items():
        cells.setdefault((fold_of[p], s), []).append(p)

    I_h, D_a, P_d = [], [], []
    for _ in range(n):
        res = []
        for _, members in cells.items():
            idx = rng.integers(0, len(members), size=len(members))
            res.extend([members[i] for i in idx])
        high = [p for p in res if stratum_of[p] == "high"]
        add = [p for p in res if stratum_of[p] == "additive"]
        if high:
            I_h.append(1.0 - _median_ratio(e_model, e_add, high))
        if add:
            D_a.append(_median_ratio(e_model, e_add, add) - 1.0)
        P_d.append(np.mean([pds_model[p] for p in res]) - np.mean([pds_add[p] for p in res]))

    def ci(v):
        if not v:
            return (float("nan"), float("nan"))
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    return {"I_high_ci": ci(I_h), "D_add_ci": ci(D_a), "pds_delta_ci": ci(P_d)}
