#!/usr/bin/env python3
"""Reject prohibited session files and coding-tool attribution.

With ``--commits RANGE`` it also checks every commit in the range: an author or
committer identity, or a message line, that names a coding tool or its vendor
fails the check, so an attributed commit is stopped at the pull request.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROHIBITED_TERMS = (
    "co" + "dex",
    "clau" + "de",
    "github " + "copilot",
)
PROHIBITED_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(term) for term in PROHIBITED_TERMS)
    + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
# Vendor domains whose addresses must never author or commit here.
PROHIBITED_IDENTITY_PATTERN = re.compile(r"@(?:" + "anthrop" + r"ic\.com|openai\.com)",
                                         re.IGNORECASE)
CONTENT_ALLOWLIST = (
    Path("agents/ops_copilot"),
    Path("tests/test_agents_ops_copilot.py"),
)


def repository_paths(root: Path) -> list[Path]:
    """Return tracked and pending non-ignored files."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [Path(item.decode()) for item in result.stdout.split(b"\0") if item]


def content_is_allowed(path: Path) -> bool:
    return any(path == allowed or allowed in path.parents for allowed in CONTENT_ALLOWLIST)


def find_violations(root: Path, paths: list[Path] | None = None) -> list[str]:
    """Return deterministic hygiene violations for repository-relative paths."""
    violations: list[str] = []
    for relative in sorted(paths if paths is not None else repository_paths(root)):
        display_path = relative.as_posix()
        if relative.name.lower().startswith("handoff-") and relative.suffix.lower() == ".md":
            violations.append(f"{display_path}: prohibited handoff file")

        absolute = root / relative
        if not absolute.is_file() or content_is_allowed(relative):
            continue
        data = absolute.read_bytes()
        if b"\0" in data:
            continue
        for line_number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
            if PROHIBITED_PATTERN.search(line):
                violations.append(
                    f"{display_path}:{line_number}: prohibited coding-tool name"
                )
    return violations


def commit_violations(root: Path, rev_range: str) -> list[str]:
    """Return attribution violations in the authors, committers and messages of a range."""
    result = subprocess.run(
        ["git", "log", "--format=%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x1e", rev_range],
        cwd=root, check=True, capture_output=True,
    )
    violations: list[str] = []
    for record in result.stdout.decode(errors="replace").split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        sha, author, committer, message = record.split("\0", 3)
        short = sha[:7]
        for role, identity in (("author", author), ("committer", committer)):
            if PROHIBITED_PATTERN.search(identity) or PROHIBITED_IDENTITY_PATTERN.search(identity):
                violations.append(f"commit {short}: prohibited {role} {identity}")
        for line in message.splitlines():
            if PROHIBITED_PATTERN.search(line) or PROHIBITED_IDENTITY_PATTERN.search(line):
                violations.append(f"commit {short}: prohibited attribution in message")
                break
    return violations


def main(argv: list[str] | None = None) -> int:
    """Check the working tree, and the commits of ``--commits RANGE`` when given."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--commits", default=None,
                        help="A revision range whose commits are checked, e.g. origin/main..HEAD.")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    violations = find_violations(root)
    if args.commits:
        violations += commit_violations(root, args.commits)
    if violations:
        print("Repository hygiene check failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
