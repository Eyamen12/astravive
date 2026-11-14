"""
ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN
Module: Project Integrity Validator & ZIP Archive Builder

USAGE:
    python package_astravive_zip.py

WHAT THIS SCRIPT DOES (AUDIT FIX: the old version listed files that did not exist and
silently skipped them):
1. Compiles every Python module (py_compile) and fails on the first syntax error.
2. Scans the HTML and Python sources for fabricated-metric and secret patterns:
   `Math.random`, Firebase / Gemini SDK strings, Google API keys (`AIzaSy...`),
   `postMessage(..., '*')`.
3. Reports every REQUIRED file that is missing and refuses to build if any are absent.
4. Packages code, data, models and docs into `ASTRAVIVE_NASA_SpaceApps_2026_Complete.zip`
   with a MANIFEST.txt listing each file and its SHA-256.
"""

from __future__ import annotations

import hashlib
import py_compile
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES: List[str] = [
    "app.py",
    "astravive_app.html",
    "astravive_data_pipeline.py",
    "astravive_ml_pipeline.py",
    "astravive_xai_engine.py",
    "astravive_digital_twin_simulator.py",
    "astravive_system_integrator.py",
    "astravive_test_suite.py",
    "package_astravive_zip.py",
    "requirements.txt",
    "README.md",
    "AUDIT_REPORT.md",
    "STEP_BY_STEP_GUIDE.md",
    "CHANGELOG.md",
]
OPTIONAL_DIRS: List[str] = ["data", "models", "docs"]

# (pattern, reason). Applied to .html/.py/.js sources only, never to this file itself.
FORBIDDEN_PATTERNS: List[Tuple[str, str]] = [
    (r"Math\.random", "unseeded randomness in UI metrics (use the seeded RNG)"),
    (r"AIzaSy[0-9A-Za-z_\-]{10,}", "Google/Firebase API key committed to source"),
    (r"firebase", "Firebase SDK injected into the app"),
    (r"generativelanguage\.googleapis|gemini", "Gemini SDK injected into the app"),
    (r"postMessage\([^)]*['\"]\*['\"]", "postMessage with wildcard target origin"),
    (r"cdn\.tailwindcss\.com", "Tailwind Play CDN is not for production; use precompiled CSS"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> List[str]:
    """Return a list of human-readable problems (empty list == clean)."""
    problems: List[str] = []
    for name in REQUIRED_FILES:
        if not (ROOT / name).exists():
            problems.append(f"MISSING required file: {name}")

    for py in sorted(ROOT.glob("*.py")):
        try:
            py_compile.compile(str(py), doraise=True)
        except py_compile.PyCompileError as exc:
            problems.append(f"SYNTAX ERROR in {py.name}: {exc.msg.strip()}")

    for src in sorted(list(ROOT.glob("*.html")) + list(ROOT.glob("*.py"))):
        if src.name in (Path(__file__).name, "astravive_test_suite.py"):
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        if text.lstrip().startswith("```"):
            problems.append(f"Stray markdown fence at top of {src.name}")
        for pattern, reason in FORBIDDEN_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                problems.append(f"FORBIDDEN in {src.name}: /{pattern}/ -> {reason}")
    return problems


def build_zip(zip_filename: str = "ASTRAVIVE_NASA_SpaceApps_2026_Complete.zip") -> Tuple[Path, List[str]]:
    files: List[Path] = [ROOT / f for f in REQUIRED_FILES]
    for d in OPTIONAL_DIRS:
        if (ROOT / d).is_dir():
            files += sorted(p for p in (ROOT / d).rglob("*") if p.is_file() and "integrated_bundle" not in p.parts
                            and p.suffix != ".db")
    zip_path = ROOT / zip_filename
    manifest_lines: List[str] = []
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            zf.write(path, f"astravive/{rel}")
            manifest_lines.append(f"{_sha256(path)}  astravive/{rel}")
        zf.writestr(
            "astravive/MANIFEST.txt",
            "ASTRAVIVE — MISSION-HEALTH DIGITAL TWIN\n"
            f"Packaged UTC: {datetime.now(timezone.utc).isoformat()}\n"
            f"Files ({len(manifest_lines)}), SHA-256:\n" + "\n".join(manifest_lines) + "\n",
        )
    return zip_path, [line.split("  ", 1)[1] for line in manifest_lines]


if __name__ == "__main__":
    print("=" * 84)
    print("ASTRAVIVE — INTEGRITY VALIDATION & PROJECT ZIP PACKAGER")
    print("=" * 84)
    issues = validate()
    if issues:
        for issue in issues:
            print(f"  [FAIL] {issue}")
        print(f"\n{len(issues)} problem(s) found. ZIP not built.")
        sys.exit(1)
    archive, included = build_zip()
    print(f"All checks passed. Packaged {len(included)} files into: {archive}")
    for item in included:
        print(f"  [OK] {item}")
