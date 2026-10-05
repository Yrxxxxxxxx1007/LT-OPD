"""Build LT-OPD with both implementations and shared evaluation tools."""
from pathlib import Path
import runpy

_hooks = runpy.run_path(str(Path(__file__).resolve().parents[2] / "_build.py"))
globals().update({name: value for name, value in _hooks.items() if not name.startswith("__")})
