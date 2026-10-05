"""LT-OPD training, inference, and evaluation."""


def load_compression_runtime(model_dir, *, implementation="auto", torch_dtype="auto", device_map="auto"):
    """Load a local export with its matching implementation.

    Use a separate Python process when loading a different implementation.
    """
    from pathlib import Path
    from .cli import _implementation_for_export, _select

    if implementation not in {"auto", "legacy", "current"}:
        raise ValueError("implementation must be 'auto', 'legacy', or 'current'")
    model_dir = Path(model_dir).expanduser().resolve()
    selected = _implementation_for_export(model_dir) if implementation == "auto" else implementation
    _select(selected)
    from training.runtime import load_cdpruner_runtime

    return load_cdpruner_runtime(model_dir, torch_dtype=torch_dtype, device_map=device_map)
