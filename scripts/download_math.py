"""Download a pinned full-MATH snapshot and export unchanged records as JSONL.

Requires pyarrow for Parquet conversion; no datasets loader or remote code runs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request
from zoneinfo import ZoneInfo

REPO = "EleutherAI/hendrycks_math"
REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
ROOT = Path(__file__).resolve().parents[1]
COUNTS = {
    "algebra": (1744, 1187),
    "counting_and_probability": (771, 474),
    "geometry": (870, 479),
    "intermediate_algebra": (1295, 903),
    "number_theory": (869, 540),
    "prealgebra": (1205, 871),
    "precalculus": (746, 546),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/math")
    args = parser.parse_args()
    import pyarrow
    import pyarrow.parquet as pq

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {output}")
    staging = output.with_name("." + output.name + "_download")
    staging.mkdir(parents=True, exist_ok=False)
    base_url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
    files = ["README.md"] + [f"{subject}/{split}-00000-of-00001.parquet"
                              for subject in COUNTS for split in ("train", "test")]

    def download(name):
        path = staging / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{base_url}/{name}", timeout=60) as response:
            path.write_bytes(response.read())
        return name, {"sha256": sha(path), "bytes": path.stat().st_size}

    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = dict(pool.map(download, files))
    manifest = {
        "dataset": "MATH", "repository": REPO, "revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
        "original_project": "https://github.com/hendrycks/math",
        "downloaded_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
        "pyarrow_version": pyarrow.__version__, "source_files": sources, "splits": {},
        "conversion": "Subject order follows COUNTS in download_math.py; original Parquet row order and all four string fields are preserved. No filtering, text normalization, IDs or extracted answers are added.",
        "download_script_sha256": sha(Path(__file__)),
    }
    for column, split in enumerate(("train", "test")):
        records, counts = [], {}
        for subject, expected in COUNTS.items():
            rows = pq.read_table(staging / "source" / subject / f"{split}-00000-of-00001.parquet").to_pylist()
            assert len(rows) == expected[column], (subject, split, len(rows))
            assert all(set(row) == {"problem", "level", "type", "solution"} and
                       all(isinstance(value, str) and value for value in row.values()) for row in rows)
            counts[subject] = len(rows)
            records.extend(rows)
        assert len(records) == (7500, 5000)[column]
        path = staging / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        assert [json.loads(line) for line in path.read_text().splitlines()] == records
        manifest["splits"][split] = {"records": len(records), "subjects": counts,
                                    "sha256": sha(path), "bytes": path.stat().st_size}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (staging / "README.md").write_text(f"""# Full MATH dataset

Downloaded from the [EleutherAI mirror]({manifest['source_url']}) of
the [original MATH dataset](https://github.com/hendrycks/math), pinned to
revision `{REVISION}`.

| Split | File | Problems |
|---|---|---:|
| Train | [train.jsonl](train.jsonl) | 7,500 |
| Test | [test.jsonl](test.jsonl) | 5,000 |

Both splits cover all seven subjects. Every record preserves the original
`problem`, `level`, `type`, and `solution` strings. JSONL conversion changes
serialization only; no rows are filtered or labels rewritten. The mirror does
not provide original per-file problem IDs, so none are invented.

Original Parquet files and the upstream dataset card remain in `source/`.
[manifest.json](manifest.json) records revision, per-subject counts, timestamps,
conversion details, and SHA-256 checksums for every downloaded and exported file.
The separate `../math500/` evaluation dataset is unchanged.

To reproduce into a new directory, install `pyarrow==21.0.0` and run
`python scripts/download_math.py --output /path/to/new/math` from the repository
root. The downloader refuses to overwrite existing data.
""")
    staging.rename(output)
    print(json.dumps({"path": str(output), "revision": REVISION,
                      "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
