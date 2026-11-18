#!/usr/bin/env bash
# Fetch the four author-provided filtered files of Norman et al. 2019
# (GEO GSE133344, ~1.1 GB total) into data/raw/. Re-running skips files
# that are already present. See SPEC.md §1 for why only these files are used.
set -euo pipefail

DEST="${1:-data/raw}"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl"
FILES=(
  GSE133344_filtered_matrix.mtx.gz
  GSE133344_filtered_genes.tsv.gz
  GSE133344_filtered_barcodes.tsv.gz
  GSE133344_filtered_cell_identities.csv.gz
)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  if [[ -s "$DEST/$f" ]]; then
    echo "have $f"
    continue
  fi
  echo "fetching $f"
  curl -fL --retry 3 -o "$DEST/$f.part" "$BASE/$f"
  mv "$DEST/$f.part" "$DEST/$f"
done

# record what was fetched so a run can be tied to exact bytes
( cd "$DEST" && sha256sum "${FILES[@]}" ) | tee "$DEST/SHA256SUMS"
