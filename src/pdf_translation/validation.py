from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ValidationConfig


VALIDATOR_REVISION = "generic-protected-literals-v1"
NUMBER_PATTERN = (
    r"(?<![\w])(?:[<>≤≥~≈]\s*)?[+−-]?\d+(?:[.,]\d+)?"
    r"(?:\s*[-–—]\s*\d+(?:[.,]\d+)?)?"
)
PLACEHOLDER_RE = re.compile(r"__PTF_(\d{4})__")


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    target: str


@dataclass(frozen=True)
class ProtectedText:
    masked: str
    replacements: tuple[tuple[str, str], ...]


def load_glossary(path: Path | None) -> tuple[GlossaryEntry, ...]:
    if path is None:
        return ()
    entries: list[GlossaryEntry] = []
    seen: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split("\t")
        if line_number == 1 and columns[0].strip().casefold() == "source":
            continue
        if len(columns) < 2 or not columns[0].strip() or not columns[1].strip():
            raise ValueError(f"Invalid glossary row at {path}:{line_number}")
        source = columns[0].strip()
        target = columns[1].strip()
        key = source.casefold()
        if key in seen and seen[key].casefold() != target.casefold():
            raise ValueError(f"Conflicting glossary entry for {source!r}")
        if key not in seen:
            entries.append(GlossaryEntry(source=source, target=target))
            seen[key] = target
    return tuple(entries)


def matching_glossary(
    source_text: str, glossary: Iterable[GlossaryEntry]
) -> tuple[GlossaryEntry, ...]:
    lowered = source_text.casefold()
    return tuple(entry for entry in glossary if entry.source.casefold() in lowered)


def _protected_matches(text: str, config: ValidationConfig) -> list[tuple[int, int]]:
    patterns: list[str] = []
    if config.protect_intervals:
        patterns.append(config.interval_pattern)
    if config.protect_numbers and config.protect_units:
        patterns.append(f"(?:{NUMBER_PATTERN})\\s*(?:{config.unit_pattern})")
    if config.protect_numbers:
        patterns.append(NUMBER_PATTERN)
    if config.protect_units:
        patterns.append(config.unit_pattern)

    candidates: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidates.append((match.start(), match.end()))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int]] = []
    cursor = -1
    for start, end in candidates:
        if start < cursor:
            continue
        selected.append((start, end))
        cursor = end
    return selected


def protect_text(text: str, config: ValidationConfig) -> ProtectedText:
    matches = _protected_matches(text, config)
    if not matches:
        return ProtectedText(masked=text, replacements=())
    pieces: list[str] = []
    replacements: list[tuple[str, str]] = []
    cursor = 0
    for index, (start, end) in enumerate(matches, start=1):
        marker = f"__PTF_{index:04d}__"
        pieces.append(text[cursor:start])
        pieces.append(marker)
        replacements.append((marker, text[start:end]))
        cursor = end
    pieces.append(text[cursor:])
    return ProtectedText(masked="".join(pieces), replacements=tuple(replacements))


def restore_text(candidate: str, protected: ProtectedText) -> tuple[str, list[dict]]:
    issues: list[dict] = []
    restored = candidate
    expected_markers = {marker for marker, _ in protected.replacements}
    observed_markers = {
        match.group(0) for match in PLACEHOLDER_RE.finditer(candidate)
    }
    if observed_markers != expected_markers:
        issues.append(
            {
                "code": "protected_placeholder_mismatch",
                "expected": sorted(expected_markers),
                "observed": sorted(observed_markers),
            }
        )
    for marker, source in protected.replacements:
        if restored.count(marker) != 1:
            issues.append(
                {
                    "code": "protected_placeholder_count",
                    "marker": marker,
                    "count": restored.count(marker),
                }
            )
        restored = restored.replace(marker, source)
    return restored, issues


def _signature(pattern: str, text: str) -> list[str]:
    return [
        re.sub(r"\s+", "", match.group(0))
        .replace(",", ".")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .casefold()
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]


def _negation_count(text: str, patterns: Iterable[str]) -> int:
    return sum(
        len(list(re.finditer(pattern, text, flags=re.IGNORECASE)))
        for pattern in patterns
    )


def validate_candidate(
    source_text: str,
    target_text: str,
    config: ValidationConfig,
    glossary: Iterable[GlossaryEntry] = (),
) -> list[dict]:
    issues: list[dict] = []
    if not target_text.strip():
        return [{"code": "empty_translation"}]

    checks: list[tuple[str, str]] = []
    if config.protect_numbers:
        checks.append(("number_signature_mismatch", NUMBER_PATTERN))
    if config.protect_units:
        checks.append(("unit_signature_mismatch", config.unit_pattern))
    if config.protect_intervals:
        checks.append(("interval_signature_mismatch", config.interval_pattern))
    for code, pattern in checks:
        source_signature = _signature(pattern, source_text)
        target_signature = _signature(pattern, target_text)
        if source_signature != target_signature:
            issues.append(
                {
                    "code": code,
                    "source": source_signature,
                    "target": target_signature,
                }
            )

    if config.negation_mode != "off":
        source_count = _negation_count(
            source_text, config.source_negation_patterns
        )
        target_count = _negation_count(
            target_text, config.target_negation_patterns
        )
        mismatch = (
            (source_count > 0) != (target_count > 0)
            if config.negation_mode == "presence"
            else source_count != target_count
        )
        if mismatch:
            issues.append(
                {
                    "code": "negation_mismatch",
                    "mode": config.negation_mode,
                    "source_count": source_count,
                    "target_count": target_count,
                }
            )

    lowered_target = target_text.casefold()
    for entry in matching_glossary(source_text, glossary):
        if entry.target.casefold() not in lowered_target:
            issues.append(
                {
                    "code": "glossary_target_missing",
                    "source_term": entry.source,
                    "target_term": entry.target,
                }
            )
    return issues
