"""Generate data/covariates.json. Run once; the result is committed."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from fertprec import covariates  # noqa: E402

typo = covariates.compute_typological_distances()
out = {
    "typo_distance": {
        "value": typo,
        "source": "URIEL via lang2vec; mean cosine distance to eng over "
                  "syntax_knn and fam (language family) feature vectors",
    },
    "data_share_proxy": {
        "value": None,
        "source": "Common Crawl published language statistics; fill from the "
                  "snapshot actually cited in the paper, not from memory",
        "note": "Deliberately left empty: a number typed in from recollection "
                "would look identical to a sourced one in the regression.",
    },
}
dest = pathlib.Path(covariates.CACHE)
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2, sort_keys=True))
print(f"wrote {dest}")
for k, v in sorted(typo.items(), key=lambda kv: kv[1]):
    print(f"  {k:<4} {v:.4f}")
