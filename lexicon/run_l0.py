"""Lexicon L0 — the canonical one-shot evaluator.

Runs all five outer folds and all five seeds before revealing any result, then
computes the mechanical verdict. Exit code IS the verdict:
    0 = PASS   1 = FAIL   2 = INDETERMINATE   3 = INVALID (manifest/spec breach)

    python -m lexicon.run_l0 --data data/proc --out runs/l0-v1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import baselines, metrics, split
from .data import load, pair_members
from .gate import decide
from .model import SEEDS, Lexicon, train_one

EXIT = {"PASS": 0, "FAIL": 1, "INDETERMINATE": 2, "INVALID": 3}


def _tensors(pairs, dbar, mean_r, device):
    di = torch.tensor(np.stack([dbar[pair_members(p)[0]] for p in pairs]), device=device)
    dj = torch.tensor(np.stack([dbar[pair_members(p)[1]] for p in pairs]), device=device)
    rt = torch.tensor(np.stack([mean_r[p] for p in pairs]), device=device)
    return di, dj, rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/proc")
    ap.add_argument("--out", default="runs/l0-v1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    ds = load(args.data)
    D, G = ds["delta"], ds["n_genes"]
    pairs, gemgroups = ds["pairs"], ds["gemgroups"]

    man_path = os.path.join(args.data, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    from .data import Manifest
    problems = Manifest(**manifest).check() if manifest else ["manifest missing"]
    manifest_ok = not problems
    if problems:
        print("[gate] MANIFEST PROBLEMS:", problems, flush=True)

    dbar = {k: split.mean_delta(D, k, gemgroups)
            for k in {kk for kk, _ in D.keys()}}
    _per_gem, mean_r = split.residuals(D, pairs, gemgroups)
    fold_of = split.assign_folds(pairs)
    centroids, thresholds, stratum_of, mag_of = split.fold_strata(mean_r, fold_of, G)
    print(f"[split] folds={np.bincount(list(fold_of.values())).tolist()} "
          f"high={sum(1 for s in stratum_of.values() if s == 'high')}", flush=True)

    lex_pred, base_pred, seed_preds = {}, {}, {s: {} for s in SEEDS[:args.seeds]}
    alphas = {}

    for f in range(split.N_FOLDS):
        train = [p for p in pairs if fold_of[p] != f]
        test = [p for p in pairs if fold_of[p] == f]
        inner = split.inner_split(train)
        fit = [p for p in train if p not in set(inner)]

        bp = baselines.predict_all(dbar, mean_r, train, inner, test, G)
        alphas[f] = bp.pop("_pca_ridge_alpha")
        for name, d in bp.items():
            base_pred.setdefault(name, {}).update(d)

        dev = args.device
        tr = _tensors(fit, dbar, mean_r, dev)
        iv = _tensors(inner, dbar, mean_r, dev)
        permitted = [k for k in dbar if "+" not in k] + fit
        delta_all = torch.tensor(np.stack([dbar[k] for k in permitted]), device=dev)
        te_i, te_j, _ = _tensors(test, dbar, mean_r, dev)

        acc = []
        for s in SEEDS[:args.seeds]:
            m, vloss = train_one(tr, iv, delta_all, G, s, device=dev)
            with torch.no_grad():
                r, _ = m.predict_residual(te_i, te_j)
            r = r.cpu().numpy()
            acc.append(r)
            for i, p in enumerate(test):
                seed_preds[s][p] = r[i]
            print(f"[fold {f}] seed {s} inner_mse={vloss:.6g} params={m.n_params()}", flush=True)
        mean_pred = np.mean(acc, axis=0)
        for i, p in enumerate(test):
            lex_pred[p] = mean_pred[i]

    # ---- scoring (nothing revealed until every fold is done) ----
    e_lex = metrics.pair_errors(mean_r, lex_pred, centroids, fold_of, G)
    e_base = {n: metrics.pair_errors(mean_r, d, centroids, fold_of, G)
              for n, d in base_pred.items()}
    pds_lex = metrics.discrimination(mean_r, lex_pred, centroids, fold_of)
    pds_add = metrics.discrimination(mean_r, base_pred["additive"], centroids, fold_of)

    summary = metrics.summarise(e_lex, e_base["additive"], pds_lex, pds_add, stratum_of)
    cis = metrics.bootstrap(e_lex, e_base["additive"], pds_lex, pds_add, stratum_of, fold_of)

    # per-baseline CI of median(e_lex / e_baseline) on the high stratum, routed
    # through the SAME frozen 10k stratified bootstrap as the primary metrics
    high = [p for p, s in stratum_of.items() if s == "high"]
    baseline_ratios = {
        n: metrics.ratio_ci_high(e_lex, eb, stratum_of, fold_of)
        for n, eb in e_base.items()
    }

    # would PCA-ridge itself clear the additive-relative gates? this — not how
    # Lexicon compares to it — decides the "weights thesis alive" redirect
    pr_pds = metrics.discrimination(mean_r, base_pred["pca_ridge"], centroids, fold_of)
    pr_sum = metrics.summarise(e_base["pca_ridge"], e_base["additive"],
                               pr_pds, pds_add, stratum_of)
    pr_cis = metrics.bootstrap(e_base["pca_ridge"], e_base["additive"],
                               pr_pds, pds_add, stratum_of, fold_of)
    pca_ridge_clears = bool(
        pr_sum["I_high"] >= 0.10 and pr_cis["I_high_ci"][0] > 0.0
        and pr_cis["D_add_ci"][1] <= 0.05
    )

    # seed-sign consistency on the primary metric
    signs = []
    for s, pr in seed_preds.items():
        es = metrics.pair_errors(mean_r, pr, centroids, fold_of, G)
        r = [es[p] / e_base["additive"][p] for p in high if e_base["additive"][p] > 0]
        signs.append(np.sign(1.0 - np.median(r)) if r else 0)
    seed_sign_consistent = len(set(signs)) <= 1

    # leave-one-gemgroup-out stability of the HIGH set specifically
    stable, stability_detail = metrics.high_set_stability(
        mean_r, D, pairs, gemgroups, fold_of, G, split)
    verdict = decide(summary, cis, baseline_ratios, manifest_ok,
                     seed_sign_consistent, stable, pca_ridge_clears)

    result = {
        "spec": "Lexicon L0 Syntax Gate v1.0",
        "summary": summary, "cis": cis,
        "baseline_ratio_ci_high": baseline_ratios,
        "baseline_median_error": {
            n: float(np.median([eb[p] for p in high])) for n, eb in e_base.items()
        },
        "lexicon_median_error_high": float(np.median([e_lex[p] for p in high])),
        "ridge_alpha_by_fold": alphas,
        "strata_stable": bool(stable), "stability_detail": stability_detail,
        "pca_ridge_clears_gates": pca_ridge_clears,
        "pca_ridge_summary": pr_sum,
        "seed_sign_consistent": bool(seed_sign_consistent),
        "manifest_ok": bool(manifest_ok), "manifest_problems": problems,
        "n_pairs": len(pairs), "n_genes": G, "seeds": list(SEEDS[:args.seeds]),
        "fold_sizes": np.bincount(list(fold_of.values())).tolist(),
        "thresholds_by_fold": {str(k): v for k, v in thresholds.items()},
        "stratum_of": stratum_of, "interaction_magnitude": mag_of,
        "runtime_sec": round(time.time() - t0, 1),
        **verdict,
    }
    # per-pair table (the gate's evidence, one row per held-out pair)
    result["pairs"] = sorted(
        [{
            "pair": p, "fold": fold_of[p], "stratum": stratum_of[p],
            "magnitude": round(mag_of[p], 6),
            "e_lexicon": round(e_lex[p], 8),
            **{f"e_{n}": round(eb[p], 8) for n, eb in e_base.items()},
            "ratio_vs_additive": round(e_lex[p] / e_base["additive"][p], 4)
            if e_base["additive"][p] > 0 else None,
            "pds_lexicon": round(pds_lex[p], 4),
        } for p in pairs],
        key=lambda r: -r["magnitude"],
    )

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # ---- POST-GATE interpretation only. Never scored, never fed back. ----
    # SPEC §3: "Program labels and pathway interpretations occur only after the
    # gate; they play no role in scoring." Fit on everything purely to read out
    # what the programs look like.
    try:
        genes = ds["genes"]
        allp = _tensors(pairs, dbar, mean_r, args.device)
        permitted_all = [k for k in dbar if "+" not in k] + pairs
        da = torch.tensor(np.stack([dbar[k] for k in permitted_all]), device=args.device)
        m, _ = train_one(allp, allp, da, G, SEEDS[0], device=args.device)
        B = m.B.detach().cpu().numpy()
        progs = []
        for k in range(B.shape[1]):
            w = B[:, k]
            top = np.argsort(-np.abs(w))[:25]
            progs.append({
                "program": k,
                "l2": float(np.linalg.norm(w)),
                "threshold": float(torch.nn.functional.softplus(m.tau[k]).item()),
                "top_genes": [{"gene": str(genes[i]).split("|")[-1],
                               "weight": round(float(w[i]), 5)} for i in top],
            })
        with open(os.path.join(args.out, "programs.json"), "w") as fh:
            json.dump({"n_programs": len(progs), "programs": progs,
                       "note": "post-gate interpretation only; never scored"}, fh, indent=2)
        print(f"[interp] wrote {len(progs)} program profiles", flush=True)
    except Exception as exc:  # interpretation must never break the verdict
        print(f"[interp] skipped: {exc}", flush=True)

    print(json.dumps({k: result[k] for k in
                      ("verdict", "reasons", "note", "summary", "cis")}, indent=2))
    code = EXIT["INVALID"] if not manifest_ok else EXIT[result["verdict"]]
    raise SystemExit(code)


if __name__ == "__main__":
    main()
