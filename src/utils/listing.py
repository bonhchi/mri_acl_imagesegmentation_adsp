"""
Helpers for handling dataset list files.

Each entry in a list file can be either:
    /absolute/path/to/volume.npz
    dataset_name|/absolute/or/relative/path/to/volume.npz

When the dataset name is omitted we attempt to infer it from the path.
All helpers normalise paths to absolute strings so downstream consumers
operate on consistent values.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List, Sequence
from collections import Counter

_KNOWN_DATASET_HINTS = {
    "fastmri",
    "kneemri",
    "oaizib",
    "oai",
    "acl",
    "combine",
    "mr",
}


@dataclass(frozen=True)
class ListEntry:
    dataset: str
    path: str


def _normalise_dataset_name(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return "unknown"
    safe = raw.replace(" ", "_")
    safe = safe.replace("-", "_")
    return safe.lower()


def infer_dataset_name(path: Path) -> str:
    """
    Heuristic inference of dataset name from a file path.
    The function walks up the directory tree searching for known hints.
    """
    parts = [p.lower() for p in path.parts]
    for token in reversed(parts):
        normalised = token.replace("-", "_")
        for hint in _KNOWN_DATASET_HINTS:
            if hint in normalised:
                return hint
    if len(parts) >= 2:
        return parts[-2].replace("-", "_")
    return parts[-1] if parts else "unknown"


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path_str = raw_path.strip()

    if os.name == "nt" and path_str.startswith("/mnt/") and len(path_str) > 6:
        drive_letter = path_str[5]
        if drive_letter.isalpha() and path_str[6] == "/":
            tail = path_str[7:].replace("/", "\\")
            path_str = f"{drive_letter.upper()}:\\{tail}"

    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def parse_list_file(list_path: str | Path) -> List[ListEntry]:
    raw_path = Path(list_path)
    # Resolve against current working directory so CLI relative paths work.
    path = raw_path if raw_path.is_absolute() else (Path.cwd() / raw_path)
    path = path.resolve()

    if not path.exists():
        search_root = path.parent
        matches: List[Path] = []
        if search_root.exists():
            matches = list(search_root.glob(f"**/{path.name}"))
        if len(matches) == 1:
            path = matches[0].resolve()
        else:
            hints = ""
            if matches:
                hints = "\n    " + "\n    ".join(str(m) for m in matches)
            message = (
                f"List file '{list_path}' not found."
                + (f" Candidates:{hints}" if hints else f" No files named '{path.name}' discovered under '{search_root}'.")
            )
            raise FileNotFoundError(message)

    base_dir = path.parent
    entries: List[ListEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            dataset_part, path_part = line.split("|", 1)
            dataset = _normalise_dataset_name(dataset_part)
        else:
            dataset = ""
            path_part = line
        resolved = _resolve_path(path_part, base_dir)
        dataset_name = dataset or infer_dataset_name(resolved)
        entries.append(ListEntry(dataset=dataset_name, path=str(resolved)))
    return entries


def format_list_entries(paths: Sequence[Path], dataset: str | None = None) -> List[str]:
    """
    Create list-file lines embedding dataset information.
    """
    dataset_name = _normalise_dataset_name(dataset or "")
    formatted: List[str] = []
    for p in paths:
        resolved = p.resolve()
        label = dataset_name or infer_dataset_name(resolved)
        formatted.append(f"{label}|{resolved}")
    return formatted


def summarise_entries(entries: Iterable[ListEntry]) -> str:
    """
    Produce a compact summary string describing dataset distribution.
    """
    counts = Counter(entry.dataset for entry in entries)
    parts = [f"{name}:{count}" for name, count in sorted(counts.items())]
    return ", ".join(parts) if parts else "empty"
