#!/usr/bin/env python3
"""Check whether the local environment is ready for dataset prep/training."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys


PACKAGES = (
    "unsloth",
    "torch",
    "datasets",
    "transformers",
    "trl",
    "bitsandbytes",
    "peft",
)


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else "no output"


def package_status(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "missing"
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return f"installed but import failed: {exc}"


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"nvidia-smi: {command_output(['nvidia-smi']) if shutil.which('nvidia-smi') else 'missing'}")
    print(f"nvcc: {command_output(['nvcc', '--version']) if shutil.which('nvcc') else 'missing'}")
    print("\nPackages:")
    for package in PACKAGES:
        print(f"- {package}: {package_status(package)}")

    if sys.version_info >= (3, 12):
        print("\nWarning: Python 3.12 may be less reliable for Unsloth than Python 3.10/3.11.")


if __name__ == "__main__":
    main()
