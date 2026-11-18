"""Lexicon L0 — synthetic unit tests (SPEC.md §6 requires these before the run).

The load-bearing one is `test_planted_grammar_is_recovered`. Without a positive
control, a FAIL verdict is ambiguous between "biology has no reusable grammar"
and "our code is broken". A gate that cannot detect a signal it planted itself
measures nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lexicon import metrics, split
from lexicon.data import CONTROL_KEY, pair_members, parse_condition
from lexicon.gate import VerdictError, decide
from lexicon.model import Lexicon, train_one


# ---- condition parser --------------------------------------------------
def test_condition_parser():
    assert parse_condition("NegCtrl0_NegCtrl0") == ("control", CONTROL_KEY)
    assert parse_condition("AHR_NegCtrl1") == ("single", "AHR")
    assert parse_condition("NegCtrl1_AHR") == ("single", "AHR")
    assert parse_condition("FEV_AHR") == ("pair", "AHR+FEV")
    assert parse_condition("AHR_FEV") == ("pair", "AHR+FEV")   # order-invariant
    assert parse_condition("") == (None, None)
    assert pair_members("AHR+FEV") == ("AHR", "FEV")


# ---- folds -------------------------------------------------------------
def test_folds_are_deterministic_and_balanced():
    pairs = [f"G{i}+H{i}" for i in range(131)]
    f1, f2 = split.assign_folds(pairs), split.assign_folds(list(reversed(pairs)))
    assert f1 == f2, "fold assignment must not depend on input order"
    sizes = sorted(np.bincount(list(f1.values())).tolist())
    assert sizes == [26, 26, 26, 26, 27], sizes


def test_inner_split_is_frozen_and_disjoint():
    train = [f"G{i}+H{i}" for i in range(100)]
    a, b = split.inner_split(train), split.inner_split(list(reversed(train)))
    assert a == b and len(a) == 10
    assert set(a).issubset(set(train))


# ---- model -------------------------------------------------------------
def test_parameter_count_is_exactly_168320():
    assert Lexicon(n_genes=5000).n_params() == 168_320


def test_additive_spine_is_structural():
    """With zero programs active the prediction must be exactly d_i + d_j."""
    m = Lexicon(n_genes=64)
    with torch.no_grad():
        m.tau.fill_(20.0)          # softplus(20) huge -> everything thresholded to 0
    d_i, d_j = torch.randn(4, 64), torch.randn(4, 64)
    r, z = m.predict_residual(d_i, d_j)
    assert torch.allclose(z, torch.zeros_like(z))
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)


def test_composition_is_symmetric():
    m = Lexicon(n_genes=64)
    d_i, d_j = torch.randn(3, 64), torch.randn(3, 64)
    r_ij, _ = m.predict_residual(d_i, d_j)
    r_ji, _ = m.predict_residual(d_j, d_i)
    assert torch.allclose(r_ij, r_ji, atol=1e-6)


# ---- planted controls (8 frozen conditions; see lexicon/controls.py) ----
# The full suite is slow, so the unit tests exercise its pieces and the
# canonical CPU + ROCm bundles are produced by `python -m lexicon.controls`.
def test_oracle_is_essentially_exact():
    """If the planted construction is not exactly recoverable from its own
    basis, the synthetic world is wrong and every other bar is moot."""
    from lexicon.controls import ORACLE_MAX, planted_world
    G, genes, dbar, pairs, mean_r, B_true, A = planted_world()
    test = pairs[680:]
    pred = np.stack([B_true @ (A[pair_members(p)[0]] * A[pair_members(p)[1]]) for p in test])
    truth = np.stack([mean_r[p] for p in test])
    u = float(((pred - truth) ** 2).mean()) / float((truth ** 2).mean())
    assert u < ORACLE_MAX, f"oracle left {u:.2e} unexplained"


@pytest.mark.slow
def test_shuffled_pairing_is_not_recovered():
    """Negative control: pairing destroyed, so there is no learnable rule and
    the model must NOT explain the targets out of sample."""
    from lexicon.controls import NEGATIVE_MIN, seed_predictions, _unexplained
    preds, tr = seed_predictions(shuffle=True, seeds=(17, 29, 43))
    u = _unexplained(torch.stack(preds).mean(0), tr)
    assert u >= NEGATIVE_MIN, f"shuffled control was 'explained' ({u:.3f} unexplained)"


@pytest.mark.slow
def test_planted_grammar_ensemble_recovers():
    from lexicon.controls import ENSEMBLE_MAX, MEDIAN_SEED_MAX, ANY_SEED_MAX
    from lexicon.controls import seed_predictions, _unexplained
    preds, tr = seed_predictions(scale=1.0)
    per_seed = [_unexplained(p, tr) for p in preds]
    ens = _unexplained(torch.stack(preds).mean(0), tr)
    assert ens <= ENSEMBLE_MAX, f"ensemble left {ens:.3f} unexplained"
    assert float(np.median(per_seed)) <= MEDIAN_SEED_MAX
    assert float(np.max(per_seed)) <= ANY_SEED_MAX


@pytest.mark.slow
def test_leave_one_seed_out_ensembles_hold():
    """Condition 6 -- proves the ensemble is not rescued by one lucky seed."""
    from lexicon.controls import LOO_ENSEMBLE_MAX, seed_predictions, _unexplained
    preds, tr = seed_predictions(scale=1.0)
    loo = [_unexplained(torch.stack([p for j, p in enumerate(preds) if j != i]).mean(0), tr)
           for i in range(len(preds))]
    assert max(loo) <= LOO_ENSEMBLE_MAX, f"leave-one-seed-out max {max(loo):.3f}"


# ---- metrics -----------------------------------------------------------
def test_additive_error_equals_interaction_magnitude():
    G = 50
    mean_r = {"a+b": np.ones(G, dtype=np.float32)}
    fold_of, centroids = {"a+b": 0}, {0: np.zeros(G, dtype=np.float32)}
    e = metrics.pair_errors(mean_r, {"a+b": np.zeros(G)}, centroids, fold_of, G)
    assert abs(e["a+b"] - 1.0) < 1e-6


def test_constant_prediction_scores_chance_discrimination():
    G, n = 20, 40
    rng = np.random.default_rng(1)
    pairs = [f"g{i}+h{i}" for i in range(n)]
    mean_r = {p: rng.normal(size=G) for p in pairs}
    fold_of = {p: 0 for p in pairs}
    centroids = {0: np.zeros(G)}
    const = {p: np.zeros(G) for p in pairs}
    pds = metrics.discrimination(mean_r, const, centroids, fold_of, n_folds=1)
    assert 0.35 < float(np.mean(list(pds.values()))) < 0.65


# ---- gate --------------------------------------------------------------
def _cis(i=(0.05, 0.30), d=(-0.02, 0.02), p=(0.02, 0.15)):
    return {"I_high_ci": i, "D_add_ci": d, "pds_delta_ci": p}


def test_gate_pass():
    s = {"n_high": 26, "I_high": 0.18, "D_add": 0.0, "pds_delta": 0.09}
    v = decide(s, _cis(), {"additive": (0.5, 0.9)}, True, True, True)
    assert v["verdict"] == "PASS"


def test_gate_fail_when_improvement_ruled_out():
    s = {"n_high": 26, "I_high": 0.01, "D_add": 0.0, "pds_delta": 0.10}
    v = decide(s, _cis(i=(-0.05, 0.04)), {"additive": (0.9, 1.0)}, True, True, True)
    assert v["verdict"] == "FAIL"


def test_gate_indeterminate_on_power_floor():
    s = {"n_high": 12, "I_high": 0.40, "D_add": 0.0, "pds_delta": 0.20}
    v = decide(s, _cis(), {"additive": (0.4, 0.8)}, True, True, True)
    assert v["verdict"] == "INDETERMINATE"
    assert "high-interaction pairs" in v["reasons"][0]


def test_gate_indeterminate_on_bad_manifest():
    s = {"n_high": 26, "I_high": 0.18, "D_add": 0.0, "pds_delta": 0.09}
    v = decide(s, _cis(), {"additive": (0.5, 0.9)}, False, True, True)
    assert v["verdict"] == "INDETERMINATE"


def test_gate_blocks_when_a_baseline_beats_us():
    s = {"n_high": 26, "I_high": 0.18, "D_add": 0.0, "pds_delta": 0.09}
    v = decide(s, _cis(), {"additive": (0.5, 0.9), "pca_ridge": (1.2, 1.6)},
               True, True, True)
    assert v["verdict"] != "PASS"
    assert "pca_ridge" in v["baselines_beating_lexicon"]


def test_weights_thesis_alive_note_fires_on_pca_ridge_clearing():
    """The redirect is decided by whether PCA-ridge ITSELF clears the gates --
    not by how Lexicon compares to it. v1.0 had this backwards."""
    s = {"n_high": 26, "I_high": 0.01, "D_add": 0.0, "pds_delta": 0.10}
    ratios = {"additive": (0.9, 1.0), "pca_ridge": (0.5, 0.8)}
    v = decide(s, _cis(i=(-0.05, 0.04)), ratios, True, True, True, pca_ridge_clears=True)
    assert v["verdict"] == "FAIL" and v["note"] == "LEXICON MODEL FAIL; WEIGHTS THESIS ALIVE"


def test_no_redirect_when_pca_ridge_also_fails():
    """If the low-rank learned baseline fails too, nothing licenses the claim
    that learned compression survived."""
    s = {"n_high": 26, "I_high": 0.01, "D_add": 0.0, "pds_delta": 0.10}
    ratios = {"additive": (0.9, 1.0), "pca_ridge": (0.5, 0.8)}
    v = decide(s, _cis(i=(-0.05, 0.04)), ratios, True, True, True, pca_ridge_clears=False)
    assert v["verdict"] == "FAIL" and v["note"] is None


def test_verdict_is_never_a_float():
    s = {"n_high": 26, "I_high": 0.18, "D_add": 0.0, "pds_delta": 0.09}
    v = decide(s, _cis(), {"additive": (0.5, 0.9)}, True, True, True)
    assert v["verdict"] in ("PASS", "FAIL", "INDETERMINATE")
    assert isinstance(v["verdict"], str)
