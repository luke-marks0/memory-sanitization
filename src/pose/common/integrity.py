from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BANNED_SHORTCUT_PATTERNS = (
    "fauxrep",
    "fauxrep2",
    "fauxrep_aux",
    "synthetic-porep",
    "window-post-fake",
)


@dataclass(frozen=True)
class BannedShortcutMatch:
    path: Path
    line_number: int
    pattern: str
    line: str


def production_scan_roots(repo_root: Path) -> tuple[Path, ...]:
    return (
        repo_root / "src" / "pose",
        repo_root / "rust" / "pose_filecoin_bridge",
    )


def excluded_scan_files(repo_root: Path) -> set[Path]:
    return {
        repo_root / "src" / "pose" / "common" / "integrity.py",
    }


def scan_production_tree_for_banned_shortcuts(repo_root: Path) -> list[BannedShortcutMatch]:
    matches: list[BannedShortcutMatch] = []
    excluded = excluded_scan_files(repo_root)
    for root in production_scan_roots(repo_root):
        for path in sorted(file_path for file_path in root.rglob("*") if file_path.is_file()):
            if path in excluded:
                continue
            if "__pycache__" in path.parts or "target" in path.parts or path.suffix == ".pyc":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line_number, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()
                for pattern in BANNED_SHORTCUT_PATTERNS:
                    if pattern in lowered:
                        matches.append(
                            BannedShortcutMatch(
                                path=path,
                                line_number=line_number,
                                pattern=pattern,
                                line=line,
                            )
                        )
    return matches


def format_matches(matches: list[BannedShortcutMatch]) -> str:
    lines = ["Banned shortcut references detected:"]
    for match in matches:
        relative = match.path
        lines.append(
            f"{relative}:{match.line_number}: {match.pattern}: {match.line}"
        )
    return "\n".join(lines)
