"""Verify both experiments on CPU; preserve every saved result and notebook output."""

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", MPLBACKEND="Agg")
os.environ.pop("PYTHONPATH", None)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/power-portability-matplotlib")
os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/power-portability-jupyter")
os.environ.setdefault("IPYTHONDIR", "/tmp/power-portability-ipython")


def check_analysis(folder):
    """Recompute summaries and figures in a temporary copy of saved results."""
    spec = importlib.util.spec_from_file_location(f"{folder.name}_analysis", folder / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((folder / "results/summary.json").read_text())
    with tempfile.TemporaryDirectory(prefix="power-analysis-") as temporary:
        module.ROOT = Path(temporary)
        module.RESULTS = module.ROOT / "results"
        module.FIGURES = module.ROOT / "figures"
        shutil.copytree(folder / "results", module.RESULTS, ignore=shutil.ignore_patterns("checkpoints"))
        with redirect_stdout(io.StringIO()):
            module.main()
        actual = json.loads((module.RESULTS / "summary.json").read_text())
        assert actual == expected, f"Analysis summary changed: {folder.name}"
        assert list(module.FIGURES.glob("*.png")), f"No figures: {folder.name}"
    import matplotlib.pyplot as plt
    plt.close("all")
    print(f"PASS: {folder.name}, regenerated analysis matches saved summary", flush=True)


def check_notebook(folder):
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    path = folder / "notebooks/results.ipynb"
    for cwd in (ROOT, folder, path.parent):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert nb.cells[0].cell_type == "markdown"
        assert "Last updated:" in nb.cells[0].source.splitlines()[0]
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv[0] = sys.executable
        try:
            NotebookClient(nb, km=manager, timeout=180,
                           resources={"metadata": {"path": str(cwd)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        print(f"PASS: {folder.name} notebook from {cwd.relative_to(ROOT)}", flush=True)


def main():
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in {
            ".venv", ".git", ".agents", ".codex", "__pycache__", ".pytest_cache",
        }]
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                assert path.resolve().is_relative_to(ROOT), f"External symlink: {path}"
            if path.suffix == ".py":
                ast.parse(path.read_text(), filename=str(path))
            if path.suffix == ".sh":
                subprocess.run(["bash", "-n", str(path)], check=True)

    for split, count in [("train", 7473), ("test", 1319)]:
        path = EXPERIMENTS / "gsm8k_qwen05b/data/gsm8k" / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == count, (split, len(rows))
        assert all("question" in row and "answer" in row for row in rows)
    math_data = EXPERIMENTS / "math500_qwen15b/data/MATH500.json"
    assert len(json.loads(math_data.read_text())) == 500
    assert hashlib.sha256(math_data.read_bytes()).hexdigest() == (
        "838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06"
    )
    print("PASS: datasets, internal symlinks, Python and shell syntax", flush=True)

    import power_sampling
    import math500_sampling
    for module in (power_sampling, math500_sampling):
        assert Path(module.__file__).resolve().is_relative_to(ROOT / "src"), module.__file__
        print(f"PASS: local import {module.__file__}", flush=True)

    # Run from outside the repository to catch accidental working-directory imports.
    with tempfile.TemporaryDirectory(prefix="power-cli-") as temporary:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "-c", str(ROOT / "pyproject.toml"), str(ROOT / "tests")],
                       cwd=temporary, check=True)
        for name, runners, report in [
            ("gsm8k_qwen05b", ["run.py", "run_vote.py"], "report.pdf"),
            ("math500_qwen15b", ["run.py"], "results_handoff.pdf"),
        ]:
            folder = EXPERIMENTS / name
            assert (folder / "reports" / report).is_file()
            for runner in runners:
                subprocess.run([sys.executable, str(folder / runner), "--help"], cwd=temporary,
                               check=True, stdout=subprocess.DEVNULL)
            print(f"PASS: {name}, CLI from outside repository", flush=True)
            check_analysis(folder)
            check_notebook(folder)
    print("PASS: CPU verification; GPU inference and fresh downloads not tested.")


if __name__ == "__main__":
    main()
