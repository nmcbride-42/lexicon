"""Lexicon L0 — the gate (SPEC.md §5).

The verdict is COMPUTED, never written. No language model may author a verdict
and no float may be reinterpreted as a judgement call. A non-enumerated verdict
raises rather than degrading to a guess.

Three outcomes: PASS / FAIL / INDETERMINATE. INDETERMINATE is a first-class
result, not a failure -- an underpowered experiment that reports its own lack
of power is behaving correctly. No further run may convert INDETERMINATE into
PASS under this spec; that requires v1.1, a new commit, and untouched data.
"""
from __future__ import annotations

MIN_HIGH_PAIRS = 20
I_HIGH_MIN = 0.10
D_ADD_MAX = 0.05
PDS_DELTA_MIN = 0.05


class VerdictError(RuntimeError):
    pass


def decide(summary: dict, cis: dict, baseline_ratios: dict,
           manifest_ok: bool, seed_sign_consistent: bool,
           strata_stable: bool, pca_ridge_clears: bool = False) -> dict:
    """Mechanical verdict. `baseline_ratios` maps baseline -> (lo, hi) CI of
    median(e_lexicon / e_baseline) on the high stratum; a lower bound above 1
    means that baseline resolvedly beats Lexicon."""
    n_high = summary["n_high"]
    I, D, P = summary["I_high"], summary["D_add"], summary["pds_delta"]
    I_lo, I_hi = cis["I_high_ci"]
    D_lo, D_hi = cis["D_add_ci"]
    P_lo, P_hi = cis["pds_delta_ci"]

    reasons, indet = [], []

    if not manifest_ok:
        indet.append("manifest/preprocessing check failed")
    if not strata_stable:
        indet.append("strata unstable under gemgroup leave-one-out")
    if not seed_sign_consistent:
        indet.append("seeds reverse the sign of the primary improvement")
    if n_high < MIN_HIGH_PAIRS:
        indet.append(f"only {n_high} high-interaction pairs (< {MIN_HIGH_PAIRS})")

    beaten_by = [b for b, (lo, _hi) in baseline_ratios.items() if lo > 1.0]

    pass_conds = {
        "n_high>=20": n_high >= MIN_HIGH_PAIRS,
        "I_high>=0.10": I >= I_HIGH_MIN,
        "I_high_lo>0": I_lo > 0.0,
        "D_add_hi<=0.05": D_hi <= D_ADD_MAX,
        "pds_delta>=0.05": P >= PDS_DELTA_MIN,
        "pds_delta_lo>0": P_lo > 0.0,
        "no_baseline_advantage": len(beaten_by) == 0,
    }
    fail_conds = {
        "I_high_hi<0.10": I_hi < I_HIGH_MIN,
        "D_add_lo>0.05": D_lo > D_ADD_MAX,
        "pds_delta_hi<0.05": P_hi < PDS_DELTA_MIN,
    }

    is_pass = all(pass_conds.values())
    is_fail = n_high >= MIN_HIGH_PAIRS and any(fail_conds.values())

    if is_pass and is_fail:
        raise VerdictError("PASS and FAIL conditions both satisfied — spec is inconsistent")

    # a CI that straddles both boundaries is not a decision
    if not is_pass and not is_fail and not indet:
        indet.append("confidence intervals straddle the pass and fail boundaries")

    if indet:
        verdict = "INDETERMINATE"
        reasons = indet
    elif is_pass:
        verdict = "PASS"
        reasons = ["all predeclared conditions met"]
    elif is_fail:
        verdict = "FAIL"
        reasons = [k for k, v in fail_conds.items() if v]
    else:
        raise VerdictError("unreachable verdict state")

    # The redirect fires when OUR architecture fails but a *learned* low-rank
    # model clears the same gates — that redirects architecture rather than
    # killing learned compression. It is decided by whether PCA-ridge itself
    # passes the additive-relative gates, NOT by how Lexicon compares to it.
    # (v1.0 had this backwards: it read "Lexicon beats PCA-ridge" as evidence
    # the thesis survived Lexicon failing, which is the opposite of the claim.)
    note = None
    if verdict == "FAIL" and pca_ridge_clears:
        note = "LEXICON MODEL FAIL; WEIGHTS THESIS ALIVE"

    if verdict not in ("PASS", "FAIL", "INDETERMINATE"):
        raise VerdictError(f"non-enumerated verdict: {verdict!r}")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "note": note,
        "pass_conditions": pass_conds,
        "fail_conditions": fail_conds,
        "baselines_beating_lexicon": beaten_by,
        "thresholds": {
            "MIN_HIGH_PAIRS": MIN_HIGH_PAIRS, "I_HIGH_MIN": I_HIGH_MIN,
            "D_ADD_MAX": D_ADD_MAX, "PDS_DELTA_MIN": PDS_DELTA_MIN,
        },
    }
