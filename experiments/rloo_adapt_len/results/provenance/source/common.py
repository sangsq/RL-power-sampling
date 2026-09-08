"""Experiment 10 protocol, files and activity records."""

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


def spec():
    return json.loads((HERE/'settings.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(message):
    with (ROOT/'agent/agent_logging').open('a') as f:
        f.write(f'\n[{stamp()}] Experiment 10: {message}\n')
    print(stamp(), message, flush=True)
