"""Lexicon L0 — the planted-control suite (SPEC.md, v1.1 amendment A10).

Eight conditions. They exist because a mechanically valid FAIL on real data is
worthless if we cannot distinguish "biology has no reusable grammar" from "the
optimiser did not converge". The suite tests THE PREDICTOR WE ACTUALLY SHIP --
the five-seed ensemble -- and proves it is not rescued by one lucky seed.

Run it on every backend you will use for the canonical run: a positive control
that never traverses the production GPU path (ROCm or CUDA) does not validate
the one-shot pipeline.

    python -m lexicon.controls --device cuda --out runs/controls-rocm.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import torch

from .data import pair_members
from .model import SEEDS, train_one

ORACLE_MAX = 1e-4
NEGATIVE_MIN = 0.90
ENSEMBLE_MAX = 0.10
MEDIAN_SEED_MAX = 0.25
ANY_SEED_MAX = 0.50
LOO_ENSEMBLE_MAX = 0.10
SCALES = (0.1, 1.0, 10.0)
SCALE_RANGE_MAX = 0.02


def planted_world(scale=1.0, shuffle=False, seed=0):
    """A world where a reusable composition rule provably exists:
    r = B_true @ (a_i * a_j). `shuffle` destroys the pairing (negative control)."""
    rng = np.random.default_rng(seed)
    G, K, n_basis = 128, 4, 40
    B_true = rng.normal(size=(G, K)) / np.sqrt(G)
    genes = [f"g{i}" for i in range(n_basis)]
    dbar = {g: (rng.normal(size=G) * 0.5 * scale).astype(np.float32) for g in genes}
    A = {g: B_true.T @ dbar[g] for g in genes}
    pairs = [f"{genes[i]}+{genes[j]}"
             for i in range(n_basis) for j in range(i + 1, n_basis)]
    truths = [(B_true @ (A[pair_members(p)[0]] * A[pair_members(p)[1]])).astype(np.float32)
              for p in pairs]
    if shuffle:
        truths = [truths[i] for i in rng.permutation(len(pairs))]
    return G, genes, dbar, pairs, dict(zip(pairs, truths)), B_true, A


def _tt(ps, dbar, mean_r, device):
    return (torch.tensor(np.stack([dbar[pair_members(p)[0]] for p in ps]), device=device),
            torch.tensor(np.stack([dbar[pair_members(p)[1]] for p in ps]), device=device),
            torch.tensor(np.stack([mean_r[p] for p in ps]), device=device))


def _unexplained(pred, truth):
    return float(((pred - truth) ** 2).mean()) / float((truth ** 2).mean())


def seed_predictions(scale=1.0, shuffle=False, device="cpu", seeds=SEEDS):
    G, genes, dbar, pairs, mean_r, _B, _A = planted_world(scale, shuffle)
    fit, inner, test = pairs[:600], pairs[600:680], pairs[680:]
    delta_all = torch.tensor(np.stack([dbar[g] for g in genes]), device=device)
    ti, tj, tr = _tt(test, dbar, mean_r, device)
    preds = []
    for s in seeds:
        m, _ = train_one(_tt(fit, dbar, mean_r, device), _tt(inner, dbar, mean_r, device),
                         delta_all, G, seed=s, device=device)
        with torch.no_grad():
            preds.append(m.predict_residual(ti, tj)[0])
    return preds, tr


def run_suite(device="cpu", verbose=True):
    """All eight frozen conditions. Returns a bundle dict with a hard `passed`."""
    def log(*a):
        if verbose:
            print(*a, flush=True)

    out = {"device": str(device), "conditions": {}, "measurements": {}}

    # 1. oracle
    G, genes, dbar, pairs, mean_r, B_true, A = planted_world()
    test = pairs[680:]
    orc = np.stack([B_true @ (A[pair_members(p)[0]] * A[pair_members(p)[1]]) for p in test])
    tru = np.stack([mean_r[p] for p in test])
    oracle_u = float(((orc - tru) ** 2).mean()) / float((tru ** 2).mean())
    out["measurements"]["oracle_unexplained"] = oracle_u
    out["conditions"]["1_oracle_lt_1e-4"] = oracle_u < ORACLE_MAX
    log(f"[1] oracle unexplained = {oracle_u:.3e}")

    # 2. shuffled negative control
    npred, ntr = seed_predictions(shuffle=True, device=device)
    neg_u = _unexplained(torch.stack(npred).mean(0), ntr)
    out["measurements"]["negative_unexplained"] = neg_u
    out["conditions"]["2_negative_gte_0.90"] = neg_u >= NEGATIVE_MIN
    log(f"[2] shuffled negative = {neg_u:.4f}")

    # 3-8. per-scale ensembles
    per_scale, ens_by_scale = {}, []
    for sc in SCALES:
        preds, tr = seed_predictions(scale=sc, device=device)
        per_seed = [_unexplained(p, tr) for p in preds]
        ens = _unexplained(torch.stack(preds).mean(0), tr)
        loo = [_unexplained(torch.stack([p for j, p in enumerate(preds) if j != i]).mean(0), tr)
               for i in range(len(preds))]
        per_scale[str(sc)] = {
            "per_seed": [round(x, 6) for x in per_seed],
            "ensemble": round(ens, 6),
            "median_seed": round(float(np.median(per_seed)), 6),
            "max_seed": round(float(np.max(per_seed)), 6),
            "leave_one_seed_out": [round(x, 6) for x in loo],
        }
        ens_by_scale.append(ens)
        log(f"[3-7] scale {sc}: ensemble={ens:.4f} median_seed={np.median(per_seed):.4f} "
            f"max_seed={np.max(per_seed):.4f} loo_max={max(loo):.4f}")

    out["measurements"]["by_scale"] = per_scale
    s1 = per_scale["1.0"]
    out["conditions"]["3_ensemble_lte_0.10_all_scales"] = all(
        per_scale[str(s)]["ensemble"] <= ENSEMBLE_MAX for s in SCALES)
    out["conditions"]["4_median_seed_lte_0.25"] = s1["median_seed"] <= MEDIAN_SEED_MAX
    out["conditions"]["5_every_seed_lte_0.50"] = s1["max_seed"] <= ANY_SEED_MAX
    out["conditions"]["6_loo_ensemble_lte_0.10_all_scales"] = all(
        max(per_scale[str(s)]["leave_one_seed_out"]) <= LOO_ENSEMBLE_MAX for s in SCALES)
    out["conditions"]["7_conditions_3_and_6_hold_at_all_scales"] = (
        out["conditions"]["3_ensemble_lte_0.10_all_scales"]
        and out["conditions"]["6_loo_ensemble_lte_0.10_all_scales"])
    rng_ = float(max(ens_by_scale) - min(ens_by_scale))
    out["measurements"]["ensemble_scale_range"] = rng_
    out["conditions"]["8_scale_range_lte_0.02"] = rng_ <= SCALE_RANGE_MAX
    log(f"[8] ensemble range across scales = {rng_:.6f}")

    out["passed"] = all(out["conditions"].values())
    out["env"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    src = b""
    for m in ("model", "data", "split", "baselines", "metrics", "gate"):
        with open(os.path.join(here, f"{m}.py"), "rb") as fh:
            src += fh.read()
    out["env"]["source_sha256"] = hashlib.sha256(src).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    b = run_suite(a.device)
    print(json.dumps({"passed": b["passed"], "conditions": b["conditions"]}, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(b, fh, indent=2)
    raise SystemExit(0 if b["passed"] else 1)


if __name__ == "__main__":
    main()
