"""Build LT-OPD from the source directory."""
from pathlib import Path
import runpy

_hooks = runpy.run_path(str(Path(__file__).resolve().parents[1] / "_build.py"))
globals().update({name: value for name, value in _hooks.items() if not name.startswith("__")})
