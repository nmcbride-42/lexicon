"""Lexicon L0 — frozen folds and interaction strata (SPEC.md §2).

Every assignment here is a deterministic function of the pair key. Nothing
depends on an outcome, a model, or a run order. The strata thresholds are set
on TRAINING pairs only; held-out interaction magnitude is used solely to file a
sealed prediction into its predeclared stratum, never to fit or select.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .data import pair_members

N_FOLDS = 5
SALT_FOLD = "lexicon-v1"
SALT_INNER = "lexicon-v1-inner"
INNER_FRAC = 0.10
STRATUM_Q = 0.80


def _digest(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()


def assign_folds(pairs, n_folds: int = N_FOLDS) -> dict:
    """Sort pairs by sha256('lexicon-v1|PAIR_KEY'), assign round-robin."""
    ordered = sorted(pairs, key=lambda p: _digest(SALT_FOLD, p))
    return {p: i % n_folds for i, p in enumerate(ordered)}


def inner_split(train_pairs, frac: float = INNER_FRAC) -> list:
    """Reserve a frozen 10% of training pairs for checkpoint selection only."""
    ordered = sorted(train_pairs, key=lambda p: _digest(SALT_INNER, p))
    n = max(1, int(round(frac * len(ordered))))
    return ordered[:n]


def mean_delta(D: dict, key: str, gemgroups) -> np.ndarray:
    return np.mean([D[(key, b)] for b in gemgroups], axis=0)


def residuals(D: dict, pairs, gemgroups):
    """r_{ij,b} = delta_ij,b - delta_i,b - delta_j,b, and its gemgroup mean."""
    per_gem, mean_r = {}, {}
    for p in pairs:
        a, b_ = pair_members(p)
        rs = []
        for g in gemgroups:
            rs.append(D[(p, g)] - D[(a, g)] - D[(b_, g)])
        per_gem[p] = np.stack(rs)
        mean_r[p] = per_gem[p].mean(axis=0)
    return per_gem, mean_r


def fold_strata(mean_r: dict, fold_of: dict, n_genes: int, n_folds: int = N_FOLDS):
    """Per fold: training centroid c_f, frozen threshold T_f, and the stratum
    of each held-out pair. Returns (centroids, thresholds, stratum_of, mag_of)."""
    centroids, thresholds, stratum_of, mag_of = {}, {}, {}, {}
    root = np.sqrt(n_genes)
    for f in range(n_folds):
        train = [p for p, ff in fold_of.items() if ff != f]
        test = [p for p, ff in fold_of.items() if ff == f]
        c_f = np.mean([mean_r[p] for p in train], axis=0)
        centroids[f] = c_f
        m_train = np.array([np.linalg.norm(mean_r[p] - c_f) / root for p in train])
        T_f = float(np.quantile(m_train, STRATUM_Q))
        thresholds[f] = T_f
        for p in test:
            m = float(np.linalg.norm(mean_r[p] - c_f) / root)
            mag_of[p] = m
            stratum_of[p] = "high" if m > T_f else "additive"
    return centroids, thresholds, stratum_of, mag_of
