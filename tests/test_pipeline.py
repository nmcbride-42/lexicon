"""End-to-end checks on a tiny synthetic pseudobulk — no download needed.

These exercise the data contract (load / manifest) and the canonical evaluator
CLI on a synthetic dataset. The synthetic manifest deliberately does NOT match
the frozen Norman et al. expectation, so the evaluator must refuse to issue a
verdict on it and exit 3 (INVALID). That is the contract working, not a bug.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest

from lexicon import split
from lexicon.data import CONTROL_KEY, EXPECTED, Manifest, load

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_synthetic(out_dir, n_genes=40, n_singles=14, n_pairs=40, gemgroups=(1, 2, 3), seed=0):
    """Pseudobulk where pair response = sum of singles + a planted product term."""
    rng = np.random.default_rng(seed)
    singles = [f"S{i:02d}" for i in range(n_singles)]
    all_pairs = [f"{a}+{b}" for i, a in enumerate(singles) for b in singles[i + 1:]]
    pairs = sorted(rng.choice(all_pairs, size=n_pairs, replace=False).tolist())
    W = rng.normal(size=(n_genes, n_genes)) / n_genes
    base = {g: rng.normal(size=n_genes) * 0.5 for g in singles}
    keys, rows = [], []
    for b in gemgroups:
        noise = lambda: rng.normal(size=n_genes) * 0.02  # noqa: E731
        for g in singles:
            keys.append(f"{g}\t{b}")
            rows.append(base[g] + noise())
        for p in pairs:
            a, c = p.split("+")
            keys.append(f"{p}\t{b}")
            rows.append(base[a] + base[c] + W @ (base[a] * base[c]) + noise())
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "pseudobulk.npz"),
        delta=np.stack(rows).astype(np.float32), keys=np.array(keys),
        genes=np.array([f"ENSG{i:05d}|GENE{i}" for i in range(n_genes)]),
        singles=np.array(singles), pairs=np.array(pairs), gemgroups=np.array(gemgroups),
    )
    man = Manifest(
        n_genes_measured=n_genes, n_identity_rows=0, n_cells_retained=0,
        n_singles=n_singles, n_pairs=n_pairs, n_gemgroups=len(gemgroups),
        gene_list_sha256="synthetic", all_pairs_have_both_singles=True,
        every_pair_in_all_gemgroups=True,
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(asdict(man), fh)
    return singles, pairs


def test_load_round_trip(tmp_path):
    singles, pairs = _write_synthetic(str(tmp_path))
    ds = load(str(tmp_path))
    assert ds["singles"] == singles and ds["pairs"] == pairs
    assert ds["gemgroups"] == [1, 2, 3] and ds["n_genes"] == 40
    assert (pairs[0], 2) in ds["delta"] and (CONTROL_KEY, 1) not in ds["delta"]


def test_manifest_check_flags_a_foreign_dataset():
    good = Manifest(**EXPECTED, gene_list_sha256="x", all_pairs_have_both_singles=True,
                    every_pair_in_all_gemgroups=True)
    assert good.check() == []
    bad = Manifest(**{**EXPECTED, "n_pairs": 130}, gene_list_sha256="x",
                   all_pairs_have_both_singles=False, every_pair_in_all_gemgroups=True)
    problems = bad.check()
    assert any("n_pairs" in p for p in problems)
    assert any("constituent single" in p for p in problems)


def test_strata_use_training_pairs_only(tmp_path):
    _singles, pairs = _write_synthetic(str(tmp_path))
    ds = load(str(tmp_path))
    _pg, mean_r = split.residuals(ds["delta"], pairs, ds["gemgroups"])
    fold_of = split.assign_folds(pairs)
    _c, thresholds, stratum_of, _m = split.fold_strata(mean_r, fold_of, ds["n_genes"])
    assert set(stratum_of) == set(pairs)          # every pair is filed exactly once
    # moving a held-out pair's outcome must not move its own fold's threshold
    p = pairs[0]
    mean_r2 = dict(mean_r)
    mean_r2[p] = mean_r[p] * 100.0
    _c2, thresholds2, _s2, _m2 = split.fold_strata(mean_r2, fold_of, ds["n_genes"])
    assert thresholds2[fold_of[p]] == thresholds[fold_of[p]]


@pytest.mark.slow
def test_evaluator_runs_end_to_end_and_refuses_a_foreign_manifest(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _write_synthetic(str(data))
    proc = subprocess.run(
        [sys.executable, "-m", "lexicon.run_l0", "--data", str(data), "--out", str(out),
         "--device", "cpu", "--seeds", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 3, proc.stdout[-2000:] + proc.stderr[-2000:]
    res = json.loads((out / "results.json").read_text())
    assert res["verdict"] in ("PASS", "FAIL", "INDETERMINATE")
    assert res["manifest_ok"] is False and res["manifest_problems"]
    assert len(res["pairs"]) == 40
    assert (out / "programs.json").exists()
