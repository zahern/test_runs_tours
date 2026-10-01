"""Prefer the local MetaCount checkout when it is available."""
import os
import sys
from pathlib import Path


DEFAULT_SOURCE = r"C:\Users\ahernz\source\MetaCount"


def activate():
    source = Path(os.environ.get("METACOUNT_SOURCE", DEFAULT_SOURCE)).expanduser()
    package_dir = source / "metacountregressor"
    if not package_dir.is_dir():
        return None
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return source_text


METACOUNT_SOURCE = activate()