"""Resolve paths recorded before the experiment layout changed."""

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
EXPERIMENT_NAMES = {
    "math500_qwen15b": "01_math500_qwen15b",
    "gsm8k_qwen05b": "02_gsm8k_qwen05b",
    "gsm8k_power_policy": "03_gsm8k_power_policy",
}


def resolve_recorded_path(path, expected_sha256=None):
    """Locate unchanged evidence without rewriting historical configurations.

    When a hash is supplied, prefer an exact frozen source snapshot over a
    refactored executable. Missing evidence or mismatched hashes raise errors.
    """
    path = Path(path)
    relative = path.relative_to(REPO) if path.is_absolute() else path
    parts = list(relative.parts)
    if parts and parts[0] == "experiments" and len(parts) > 1:
        parts[1] = EXPERIMENT_NAMES.get(parts[1], parts[1])
    current = REPO.joinpath(*parts)
    if len(parts) > 2 and parts[2] == "data":
        if parts[1] == "02_gsm8k_qwen05b":
            current = DATA.joinpath(*parts[3:])
        elif parts[1] == "01_math500_qwen15b":
            current = DATA / "math500" / Path(*parts[3:])
    candidates = [REPO / relative, current]
    if len(parts) > 2 and parts[0] == "experiments":
        folder = REPO / "experiments" / parts[1]
        tail = Path(*parts[2:])
        candidates.extend([folder / "scripts" / tail, folder / "results" / tail,
                           folder / "figures" / tail, folder / "results/archive" / tail])
        if expected_sha256:
            candidates.extend(sorted((folder / "results").glob(f"**/source/**/{path.name}")))
    for candidate in candidates:
        if not candidate.exists():
            continue
        if expected_sha256 is None or (candidate.is_file() and
                hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_sha256):
            return candidate
    raise FileNotFoundError(f"Cannot resolve recorded evidence: {path}")
