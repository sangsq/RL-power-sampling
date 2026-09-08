# Full MATH dataset

Downloaded from the [EleutherAI mirror](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/21a5633873b6a120296cce3e2df9d5550074f4a3) of
the [original MATH dataset](https://github.com/hendrycks/math), pinned to
revision `21a5633873b6a120296cce3e2df9d5550074f4a3`.

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
