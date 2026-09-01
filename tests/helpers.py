from __future__ import annotations

import json
from pathlib import Path


def write_profile(
    root: Path,
    *,
    source_pdf: Path,
    expected_pages: int | None = None,
    negation_mode: str = "off",
) -> Path:
    profiles = root / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    glossary = profiles / "glossary.tsv"
    glossary.write_text("source\ttarget\n", encoding="utf-8")
    profile = {
        "schema_version": "1.0",
        "project": {
            "id": "test",
            "source_pdf": str(source_pdf),
            "workspace": str(root / "workspace"),
            "release_dir": str(root / "release"),
        },
        "document": {
            "expected_sha256": None,
            "expected_pages": expected_pages,
            "extract_tables": False,
        },
        "translation": {
            "source_language": "English",
            "target_language": "Russian",
            "prompt_template": (
                "Translate {source_language} to {target_language}. "
                "Preserve placeholders. Return only the translation."
            ),
            "glossary": str(glossary),
            "primary": {
                "kind": "openai_chat",
                "endpoint": "http://127.0.0.1:1/v1/chat/completions",
                "model": "primary",
            },
            "fallback": {
                "kind": "openai_chat",
                "endpoint": "http://127.0.0.1:2/v1/chat/completions",
                "model": "fallback",
            },
        },
        "validation": {
            "protect_numbers": True,
            "protect_units": True,
            "protect_intervals": True,
            "negation_mode": negation_mode,
            "source_negation_patterns": [
                "\\b(?:no|not|never|without)\\b"
            ],
            "target_negation_patterns": ["\\b(?:не|нет|никогда|без)\\b"],
            "unit_pattern": "\\b(?:mg|g|kg|mL|L|h)\\b",
            "interval_pattern": "\\b(?:q\\s*\\d+\\s*h|BID|PRN)\\b",
        },
    }
    path = profiles / "test.json"
    path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
