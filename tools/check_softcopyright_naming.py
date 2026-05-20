#!/usr/bin/env python
"""Check naming consistency for software copyright materials."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def read_text(path: Path) -> str:
    encodings = ("utf-8-sig", "utf-8", "gbk", "utf-16")
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # Last resort to avoid hard crash on mixed files.
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "docs" / "softcopyright" / "naming_manifest.json"

    if not manifest_path.exists():
        print(f"[FAIL] Missing manifest: {manifest_path}")
        return 1

    data = json.loads(read_text(manifest_path))
    main_name = data["main_name"]
    alternate_names = data.get("alternate_names", [])
    subtitle = data["subtitle"]
    abstract_first_sentence = data["abstract_first_sentence"]
    keywords = data.get("keywords", [])
    version = data["version"]
    targets = data.get("check_targets", [])

    failures: list[str] = []
    all_text_chunks: list[str] = []

    for rel in targets:
        path = root / rel
        if not path.exists():
            failures.append(f"Missing target file: {rel}")
            continue

        text = read_text(path)
        all_text_chunks.append(text)

        if main_name not in text:
            failures.append(f"Main name missing in {rel}")

        if version not in text:
            failures.append(f"Version '{version}' missing in {rel}")

        for alt in alternate_names:
            if alt in text:
                failures.append(f"Alternate name appears in target file {rel}: {alt}")

        if re.search(r"\bv1(?!\.0)\b", text, flags=re.IGNORECASE):
            failures.append(f"Found forbidden version style 'v1' in {rel}")
        if "v1.0" in text:
            failures.append(f"Found lowercase version style 'v1.0' in {rel}")
        if "1.0.0" in text:
            failures.append(f"Found forbidden version style '1.0.0' in {rel}")

    combined = "\n".join(all_text_chunks)

    if subtitle not in combined:
        failures.append("Subtitle is missing across all target materials")

    if abstract_first_sentence not in combined:
        failures.append("Abstract first sentence is missing across all target materials")

    for kw in keywords:
        if kw not in combined:
            failures.append(f"Keyword missing across all target materials: {kw}")

    if failures:
        print("[FAIL] Naming consistency check failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("[PASS] Naming consistency check passed.")
    print(f"- Main name: {main_name}")
    print(f"- Version: {version}")
    print(f"- Checked files: {len(targets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

