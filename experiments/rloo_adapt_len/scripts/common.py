"""Adaptive-length RLOO protocol, files and activity records."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PYTHON = ROOT/'.venv/bin/python'


def stamp():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')


def rows(path):
    return [json.loads(l) for l in Path(path).open()] if Path(path).exists() else []


def settings_path(alpha=2):
    return HERE/('settings.json' if alpha == 2 else f'settings_alpha{alpha:g}.json')


def spec(alpha=2):
    return json.loads(settings_path(alpha).read_text())


def run_name(alpha):
    return f'alpha{alpha:g}_adapt_len'


def evaluation_path(alpha):
    return HERE/'results'/('evaluation' if alpha == 2 else f'evaluation_alpha{alpha:g}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(message):
    (ROOT/'agent').mkdir(exist_ok=True)
    with (ROOT/'agent/agent_logging').open('a') as f:
        f.write(f'\n[{stamp()}] Adaptive-length RLOO: {message}\n')
    print(stamp(), message, flush=True)
