"""Use the repository build configuration from either source directory."""
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path

from setuptools import build_meta

ROOT = Path(__file__).resolve().parent


@contextmanager
def repository_directory():
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def repository_hook(hook):
    @wraps(hook)
    def call(*args, **kwargs):
        with repository_directory():
            return hook(*args, **kwargs)
    return call


for _name in (
    "get_requires_for_build_wheel", "get_requires_for_build_editable",
    "get_requires_for_build_sdist", "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable", "build_wheel", "build_editable",
    "build_sdist",
):
    globals()[_name] = repository_hook(getattr(build_meta, _name))
