"""Lexicon L0 — data pipeline for Norman et al. 2019 CRISPRa Perturb-seq (GSE133344).

Implements SPEC.md §1 exactly. The manifest check is the contract: if the
numbers this module reports do not match the frozen expectation, the run is
INVALID — we do not proceed on a dataset we cannot identify.

Modelling unit is condition x gemgroup pseudobulk, never individual cells.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---- frozen expectation (SPEC.md §1 / §6) -------------------------------
EXPECTED = {
    "n_genes_measured": 33694,
    "n_identity_rows": 111445,
    "n_cells_retained": 91168,
    "n_singles": 105,
    "n_pairs": 131,
    "n_gemgroups": 8,
}
N_HVG = 5000
CONTROL_KEY = "__control__"


# ---- condition parsing (SPEC.md §1 preprocessing step 2) ----------------
def _is_ctrl(tok: str) -> bool:
    return tok.strip().lower().startswith("negctrl")


def parse_condition(guide_identity: str):
    """guide_identity -> (kind, key).

    kind is one of 'control' | 'single' | 'pair'; key is CONTROL_KEY, the gene
    name, or the canonical sorted 'A+B' pair key.

        gene + NegCtrl   -> single gene
        geneA + geneB    -> sorted unordered pair
        NegCtrl+NegCtrl  -> control
    """
    if not isinstance(guide_identity, str) or not guide_identity.strip():
        return None, None
    s = guide_identity.strip()
    # GSE133344 doubles the guide identity: "ARID1A_NegCtrl0__ARID1A_NegCtrl0".
    # The two halves are the same assignment; take the first.
    if "__" in s:
        s = s.split("__")[0]
    toks = [t for t in s.split("_") if t != ""]
    if len(toks) != 2:
        return None, None
    a, b = toks
    ca, cb = _is_ctrl(a), _is_ctrl(b)
    if ca and cb:
        return "control", CONTROL_KEY
    if ca:
        return "single", b
    if cb:
        return "single", a
    if a == b:
        return "single", a
    lo, hi = sorted([a, b])
    return "pair", f"{lo}+{hi}"


def pair_members(pair_key: str):
    return tuple(pair_key.split("+"))


# ---- raw IO ------------------------------------------------------------
def _read_mtx(path: str):
    """Stream a MatrixMarket coordinate file into CSR. Faster and far more
    memory-predictable than scipy.io.mmread on a 1.1GB gz."""
    with gzip.open(path, "rt") as fh:
        line = fh.readline()
        if not line.startswith("%%MatrixMarket"):
            raise ValueError(f"not a MatrixMarket file: {path}")
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line.startswith("%"):
                break
        nrows, ncols, nnz = (int(x) for x in line.split())
        tri = pd.read_csv(
            fh, sep=r"\s+", header=None, names=["r", "c", "v"],
            dtype={"r": np.int32, "c": np.int32, "v": np.float32}, engine="c",
        )
    if len(tri) != nnz:
        raise ValueError(f"nnz mismatch: header {nnz}, read {len(tri)}")
    m = sp.csr_matrix(
        (tri["v"].to_numpy(), (tri["r"].to_numpy() - 1, tri["c"].to_numpy() - 1)),
        shape=(nrows, ncols),
    )
    return m


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    n_genes_measured: int
    n_identity_rows: int
    n_cells_retained: int
    n_singles: int
    n_pairs: int
    n_gemgroups: int
    gene_list_sha256: str
    all_pairs_have_both_singles: bool
    every_pair_in_all_gemgroups: bool

    def check(self) -> list:
        problems = []
        for k, v in EXPECTED.items():
            got = getattr(self, k)
            if got != v:
                problems.append(f"{k}: expected {v}, got {got}")
        if not self.all_pairs_have_both_singles:
            problems.append("some pair is missing a constituent single")
        if not self.every_pair_in_all_gemgroups:
            problems.append("some pair is not present in all 8 gemgroups")
        return problems


# ---- the pipeline ------------------------------------------------------
def build(raw_dir: str, out_dir: str, n_hvg: int = N_HVG, verbose: bool = True):
    """Raw GEO files -> pseudobulk deltas + manifest. Writes out_dir/pseudobulk.npz."""
    def log(*a):
        if verbose:
            print(*a, flush=True)

    f_mtx = os.path.join(raw_dir, "GSE133344_filtered_matrix.mtx.gz")
    f_gene = os.path.join(raw_dir, "GSE133344_filtered_genes.tsv.gz")
    f_bc = os.path.join(raw_dir, "GSE133344_filtered_barcodes.tsv.gz")
    f_id = os.path.join(raw_dir, "GSE133344_filtered_cell_identities.csv.gz")

    genes = pd.read_csv(f_gene, sep="\t", header=None)
    gene_ids = genes.iloc[:, 0].astype(str).to_numpy()
    gene_names = (genes.iloc[:, 1] if genes.shape[1] > 1 else genes.iloc[:, 0]).astype(str).to_numpy()
    barcodes = pd.read_csv(f_bc, sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()
    ident = pd.read_csv(f_id)
    log(f"[data] genes={len(gene_ids)} barcodes={len(barcodes)} identity_rows={len(ident)}")

    log("[data] reading matrix (streaming triplets) ...")
    X = _read_mtx(f_mtx)
    # orient to cells x genes
    if X.shape[0] == len(gene_ids) and X.shape[1] == len(barcodes):
        X = X.T.tocsr()
    elif not (X.shape[0] == len(barcodes) and X.shape[1] == len(gene_ids)):
        raise ValueError(f"matrix shape {X.shape} matches neither orientation")
    log(f"[data] matrix cells x genes = {X.shape}, nnz={X.nnz}")

    bc_col = "cell_barcode" if "cell_barcode" in ident.columns else ident.columns[0]
    ident[bc_col] = ident[bc_col].astype(str)

    # SPEC §1 step 1: retain high-confidence singlets only
    keep = ident.copy()
    if "good_coverage" in keep.columns:
        gc = keep["good_coverage"]
        keep = keep[gc.astype(str).str.lower().isin(["true", "1", "1.0"])]
    if "number_of_cells" in keep.columns:
        keep = keep[pd.to_numeric(keep["number_of_cells"], errors="coerce") == 1]
    log(f"[data] retained cells after singlet/coverage filter = {len(keep)}")

    kinds, keys = zip(*[parse_condition(g) for g in keep["guide_identity"]])
    keep = keep.assign(_kind=kinds, _key=keys)
    keep = keep[keep["_kind"].notna()]

    bc_to_row = {b: i for i, b in enumerate(barcodes)}
    keep = keep[keep[bc_col].isin(bc_to_row)]
    rows = keep[bc_col].map(bc_to_row).to_numpy()
    gem = pd.to_numeric(keep["gemgroup"], errors="coerce").to_numpy()

    singles = sorted({k for kd, k in zip(keep["_kind"], keep["_key"]) if kd == "single"})
    pairs = sorted({k for kd, k in zip(keep["_kind"], keep["_key"]) if kd == "pair"})
    gemgroups = sorted({int(g) for g in gem if not np.isnan(g)})
    log(f"[data] singles={len(singles)} pairs={len(pairs)} gemgroups={gemgroups}")

    single_set = set(singles)
    both = all(a in single_set and b in single_set for a, b in map(pair_members, pairs))

    # SPEC §1 step 4: HVG on control+single cells ONLY, never doubles.
    # seurat_v3 is defined on raw counts, so it is applied to counts here and
    # the resulting gene set is used for the lognorm pseudobulk (see SPEC note).
    import scanpy as sc
    from anndata import AnnData

    mask_cs = keep["_kind"].isin(["control", "single"]).to_numpy()
    log(f"[data] HVG selection on {int(mask_cs.sum())} control+single cells ...")
    ad = AnnData(X=X[rows[mask_cs]].astype(np.float32))
    ad.obs["gemgroup"] = pd.Categorical(gem[mask_cs].astype(int).astype(str))
    ad.var_names = pd.Index([f"{i}|{n}" for i, n in zip(gene_ids, gene_names)])
    sc.pp.highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=n_hvg, batch_key="gemgroup")
    hvg_idx = np.where(ad.var["highly_variable"].to_numpy())[0]
    hvg_idx.sort()
    del ad
    log(f"[data] selected {len(hvg_idx)} HVGs")

    # SPEC §1 step 3: per-cell library-size normalise to 10k then log1p
    log("[data] normalising ...")
    Xk = X[rows][:, hvg_idx].astype(np.float32)
    lib = np.asarray(X[rows].sum(axis=1)).ravel()      # full library, not HVG-only
    lib[lib == 0] = 1.0
    Xk = sp.diags((1e4 / lib).astype(np.float32)) @ Xk
    Xk.data = np.log1p(Xk.data)

    # SPEC §1 step 5/6: condition x gemgroup pseudobulk, delta vs control
    cond_key = keep["_key"].to_numpy()
    gem_i = gem.astype(int)
    means, index = {}, []
    for b in gemgroups:
        sel_b = gem_i == b
        for k in np.unique(cond_key[sel_b]):
            sel = sel_b & (cond_key == k)
            means[(k, b)] = np.asarray(Xk[sel].mean(axis=0)).ravel()
            index.append((k, b))

    deltas, rec = {}, []
    for (k, b) in index:
        ctrl = means.get((CONTROL_KEY, b))
        if ctrl is None or k == CONTROL_KEY:
            continue
        deltas[(k, b)] = (means[(k, b)] - ctrl).astype(np.float32)
        rec.append((k, b))

    every_pair_all_gem = all(
        all((p, b) in deltas for b in gemgroups) for p in pairs
    )

    gene_list = np.array([f"{gene_ids[i]}|{gene_names[i]}" for i in hvg_idx])
    man = Manifest(
        n_genes_measured=len(gene_ids),
        n_identity_rows=len(ident),
        n_cells_retained=int(len(keep)),
        n_singles=len(singles),
        n_pairs=len(pairs),
        n_gemgroups=len(gemgroups),
        gene_list_sha256=hashlib.sha256("\n".join(gene_list).encode()).hexdigest(),
        all_pairs_have_both_singles=bool(both),
        every_pair_in_all_gemgroups=bool(every_pair_all_gem),
    )

    os.makedirs(out_dir, exist_ok=True)
    keys_arr = np.array([f"{k}\t{b}" for k, b in rec])
    mat = np.stack([deltas[(k, b)] for k, b in rec]).astype(np.float32)
    np.savez_compressed(
        os.path.join(out_dir, "pseudobulk.npz"),
        delta=mat, keys=keys_arr, genes=gene_list,
        singles=np.array(singles), pairs=np.array(pairs),
        gemgroups=np.array(gemgroups),
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(asdict(man), fh, indent=2)
    log(f"[data] wrote pseudobulk {mat.shape} -> {out_dir}")
    return man


def load(out_dir: str):
    """Load pseudobulk into dicts keyed by (condition, gemgroup)."""
    z = np.load(os.path.join(out_dir, "pseudobulk.npz"), allow_pickle=False)
    delta, keys = z["delta"], z["keys"]
    D = {}
    for i, kb in enumerate(keys):
        k, b = str(kb).split("\t")
        D[(k, int(b))] = delta[i]
    return {
        "delta": D,
        "genes": z["genes"],
        "singles": [str(s) for s in z["singles"]],
        "pairs": [str(p) for p in z["pairs"]],
        "gemgroups": [int(g) for g in z["gemgroups"]],
        "n_genes": delta.shape[1],
    }


def main(argv=None):
    """CLI: raw GEO files -> pseudobulk + manifest. Exit 0 only if the manifest
    matches the frozen expectation.

        python -m lexicon.data --raw data/raw --out data/proc
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw", help="directory holding the four GSE133344 files")
    ap.add_argument("--out", default="data/proc", help="where to write pseudobulk.npz + manifest.json")
    a = ap.parse_args(argv)
    problems = build(a.raw, a.out).check()
    print(json.dumps({"manifest_ok": not problems, "problems": problems}, indent=2))
    raise SystemExit(0 if not problems else 3)


if __name__ == "__main__":
    main()
