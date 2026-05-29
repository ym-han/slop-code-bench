"""CQB-commensurable diff stats via git + diffstat.

Produces numbers byte-for-byte commensurable with CQB's `computeScores` pipeline
(harness/lib/utils/git.ts + harness/lib/utils/diffstat.ts).  Shell out to the same
`git` / `diffstat` binaries rather than reimplementing the semantics in Python.

Requirements (host):
    git  >= 2.x  (--no-index, -M, --ignore-space-change)
    diffstat >= 1.x  (-t -m flags for CSV + modification-aware output)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from slop_code.logging import get_logger

logger = get_logger(__name__)

_EXPECTED_HEADER = "INSERTED,DELETED,MODIFIED,FILENAME"


@dataclass(frozen=True)
class FileStatEntry:
    filename: str
    insertions: int
    deletions: int
    modifications: int


@dataclass(frozen=True)
class DiffstatResult:
    files_changed: int
    lines_changed: int
    per_file: list[FileStatEntry]

    def to_dict(self) -> dict:
        return {
            "files_changed": self.files_changed,
            "lines_changed": self.lines_changed,
            "per_file": [
                {
                    "filename": e.filename,
                    "insertions": e.insertions,
                    "deletions": e.deletions,
                    "modifications": e.modifications,
                }
                for e in self.per_file
            ],
        }


_EMPTY = DiffstatResult(files_changed=0, lines_changed=0, per_file=[])


def _parse_file_stat_line(line: str) -> FileStatEntry:
    # Format: INSERTED,DELETED,MODIFIED,FILENAME (filename may contain commas)
    parts = line.split(",")
    if len(parts) < 4:
        raise ValueError(f"Invalid CSV line: {line!r}")
    insertions = int(parts[0])
    deletions = int(parts[1])
    modifications = int(parts[2])
    filename_raw = ",".join(parts[3:])
    filename = filename_raw.strip('"')
    return FileStatEntry(
        filename=filename,
        insertions=insertions,
        deletions=deletions,
        modifications=modifications,
    )


def parse_diffstat_output(output: str) -> DiffstatResult:
    """Parse output of `git diff … | diffstat -tm`.

    Mirrors CQB's parseDiffstatOutput (harness/lib/utils/diffstat.ts).
    Returns an empty result when the diff is empty.
    """
    trimmed = output.strip()
    if not trimmed:
        return _EMPTY

    lines = trimmed.splitlines()
    header = lines[0].strip()
    if header != _EXPECTED_HEADER:
        raise ValueError(
            f"Invalid diffstat output format. Expected header {_EXPECTED_HEADER!r}, "
            f"got {header!r}"
        )

    per_file = [_parse_file_stat_line(ln) for ln in lines[1:] if ln.strip()]
    lines_changed = sum(e.insertions + e.deletions + e.modifications for e in per_file)
    return DiffstatResult(
        files_changed=len(per_file),
        lines_changed=lines_changed,
        per_file=per_file,
    )


def git_diffstat_between_dirs(from_dir: Path | None, to_dir: Path) -> DiffstatResult:
    """Compute CQB-commensurable diff stats between two snapshot directories.

    Uses `git diff --no-index -M --ignore-space-change | diffstat -tm`, matching
    the exact flags in CQB's git.ts / diffstat.ts so numbers are comparable.

    Args:
        from_dir: The "before" directory.  None is treated as an empty baseline
            (all files in to_dir are counted as insertions).
        to_dir: The "after" directory.

    Returns:
        A DiffstatResult with files_changed, lines_changed, and per-file breakdown.
    """
    import tempfile

    if from_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return _run_git_diffstat(Path(tmp), to_dir)
    return _run_git_diffstat(from_dir, to_dir)


def _run_git_diffstat(from_dir: Path, to_dir: Path) -> DiffstatResult:
    """Run git diff --no-index | diffstat and parse the result."""
    git_cmd = [
        "git",
        "diff",
        "--no-index",
        "--no-prefix",
        "-M",
        "--ignore-space-change",
        str(from_dir),
        str(to_dir),
    ]

    git_proc = subprocess.run(
        git_cmd,
        capture_output=True,
        text=True,
        # git diff --no-index exits 1 when there are differences — not an error
    )
    if git_proc.returncode not in (0, 1):
        logger.warning(
            "git diff --no-index failed",
            returncode=git_proc.returncode,
            stderr=git_proc.stderr,
        )
        return _EMPTY

    diff_text = git_proc.stdout

    diffstat_proc = subprocess.run(
        ["diffstat", "-t", "-m"],
        input=diff_text,
        capture_output=True,
        text=True,
    )
    if diffstat_proc.returncode != 0:
        logger.warning(
            "diffstat failed",
            returncode=diffstat_proc.returncode,
            stderr=diffstat_proc.stderr,
        )
        return _EMPTY

    return parse_diffstat_output(diffstat_proc.stdout)
