"""Run LT-OPD with the selected implementation."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import runpy
import sys


MODULES = {
    "train": "training.train",
    "export": "training.export",
    "data": "data.prepare",
    "eval": "evaluation.run",
    "score": "evaluation.score",
}


def _select(implementation):
    shared = Path(__file__).resolve().parents[1]
    selected = shared if implementation == "legacy" else shared / "learnable_merge"
    if not (selected / "training" / "__init__.py").is_file():
        raise ImportError(f"Missing LT-OPD implementation: {implementation}")
    for name in ("training", "verl"):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != selected / name:
            raise RuntimeError(f"{name} is already imported from another implementation; start a new Python process")
    paths = list(dict.fromkeys((str(selected), str(shared))))
    sys.path[:] = paths + [item for item in sys.path if item not in paths]
    existing = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths + [item for item in existing if item]))
    os.environ["LT_OPD_IMPLEMENTATION"] = implementation


def _run(command, implementation, arguments):
    _select(implementation)
    sys.argv = [f"lt-opd {command}", *arguments]
    runpy.run_module(MODULES[command], run_name="__main__")


def main():
    parser = argparse.ArgumentParser(
        prog="lt-opd",
        description=__doc__,
        epilog=("Code: https://github.com/Yrxxxxxxxx1007/LT-OPD\n"
                "Model: https://huggingface.co/yyy051007/LT-OPD"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    try:
        release = version("lt-opd")
    except PackageNotFoundError:
        release = "source"
    parser.add_argument("--version", action="version", version=f"%(prog)s {release}")
    parser.add_argument("--implementation", choices=("legacy", "current"), default="legacy",
                        help="Implementation to use (default: legacy)")
    parser.add_argument("command", choices=tuple(MODULES))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    _run(args.command, args.implementation, args.arguments)


def train():
    _run("train", "legacy", sys.argv[1:])


def export():
    _run("export", "legacy", sys.argv[1:])


def data():
    _run("data", "legacy", sys.argv[1:])


def evaluate():
    _run("eval", "legacy", sys.argv[1:])


def score():
    _run("score", "legacy", sys.argv[1:])


def merge_train():
    _run("train", "current", sys.argv[1:])


def merge_export():
    _run("export", "current", sys.argv[1:])
