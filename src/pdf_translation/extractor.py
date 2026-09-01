from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import pymupdf as fitz

from .config import Profile
from .io import (
    atomic_write_json,
    atomic_write_jsonl,
    fingerprint,
    load_json,
    sha256_file,
    sha256_text,
)


EXTRACTOR_REVISION = "generic-pymupdf-extractor-v1"
PAGE_FILE_RE = re.compile(r"page_(\d{4,})\.json$")


class ExtractionError(RuntimeError):
    """Source extraction could not preserve the declared invariants."""


def parse_pages(specification: str | None, total_pages: int) -> tuple[int, ...]:
    if not specification or specification.strip().casefold() == "all":
        return tuple(range(1, total_pages + 1))
    selected: set[int] = set()
    for raw in specification.split(","):
        item = raw.strip()
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", item)
        if match is None:
            raise ExtractionError(f"Invalid page selector: {item!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start or end > total_pages:
            raise ExtractionError(
                f"Page selector {item!r} is outside 1-{total_pages}"
            )
        selected.update(range(start, end + 1))
    return tuple(sorted(selected))


def _round_rect(value: Iterable[float]) -> list[float]:
    return [round(float(item), 3) for item in value]


def _overlap_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    a = fitz.Rect(first)
    b = fitz.Rect(second)
    intersection = a & b
    if intersection.is_empty or a.get_area() <= 0:
        return 0.0
    return float(intersection.get_area() / a.get_area())


def _span_payload(span: dict[str, Any]) -> dict[str, Any]:
    color = int(span.get("color", 0) or 0)
    return {
        "text": str(span.get("text") or ""),
        "bbox": _round_rect(span.get("bbox", (0, 0, 0, 0))),
        "font": str(span.get("font") or ""),
        "size": round(float(span.get("size", 0.0) or 0.0), 3),
        "flags": int(span.get("flags", 0) or 0),
        "color": f"#{color & 0xFFFFFF:06X}",
        "alpha": int(span.get("alpha", 255) or 0),
    }


def _block_text(block: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    spans: list[dict[str, Any]] = []
    for line in block.get("lines", []):
        line_spans = [_span_payload(span) for span in line.get("spans", [])]
        text = "".join(span["text"] for span in line_spans).strip()
        if text:
            lines.append(text)
        spans.extend(line_spans)
    return "\n".join(lines).strip(), spans


def _extract_tables(page: fitz.Page) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    try:
        finder = page.find_tables()
    except Exception as exc:
        return [], [f"table_detection_failed:{type(exc).__name__}:{exc}"]
    tables: list[dict[str, Any]] = []
    for table_index, table in enumerate(getattr(finder, "tables", ()), start=1):
        table_id = f"p{page.number + 1:04d}-t{table_index:03d}"
        extracted = table.extract()
        row_count = int(getattr(table, "row_count", len(extracted)))
        column_count = int(
            getattr(
                table,
                "col_count",
                max((len(row) for row in extracted), default=0),
            )
        )
        raw_cells = list(getattr(table, "cells", ()))
        cells: list[dict[str, Any]] = []
        for row in range(row_count):
            for column in range(column_count):
                source_text = ""
                if row < len(extracted) and column < len(extracted[row]):
                    source_text = str(extracted[row][column] or "").strip()
                flat_index = row * column_count + column
                raw_bbox = (
                    raw_cells[flat_index]
                    if flat_index < len(raw_cells) and raw_cells[flat_index] is not None
                    else (0, 0, 0, 0)
                )
                unit_id = f"{table_id}-r{row + 1:03d}-c{column + 1:03d}"
                cells.append(
                    {
                        "unit_id": unit_id,
                        "row": row,
                        "column": column,
                        "bbox": _round_rect(raw_bbox),
                        "source_text": source_text,
                        "source_sha256": sha256_text(source_text),
                        "role": "table_cell",
                    }
                )
        tables.append(
            {
                "table_id": table_id,
                "bbox": _round_rect(table.bbox),
                "row_count": row_count,
                "column_count": column_count,
                "cells": cells,
            }
        )
    return tables, warnings


def _security_audit(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    suspicious: list[dict[str, Any]] = []
    for block in blocks:
        for span in block.get("spans", []):
            reasons: list[str] = []
            if 0 < float(span.get("size", 0)) < 2:
                reasons.append("very_small_text")
            if int(span.get("alpha", 255)) < 32:
                reasons.append("nearly_transparent_text")
            if str(span.get("color", "")).upper() in {"#FFFFFF", "#FEFEFE", "#FDFDFD"}:
                reasons.append("near_white_text")
            if reasons and str(span.get("text") or "").strip():
                suspicious.append(
                    {
                        "unit_id": block["unit_id"],
                        "text_excerpt": str(span["text"])[:160],
                        "reasons": reasons,
                    }
                )
    return {
        "status": "warning" if suspicious else "pass",
        "suspicious_span_count": len(suspicious),
        "suspicious_spans": suspicious,
    }


def iter_source_units(page: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for block in page.get("blocks", []):
        if str(block.get("source_text") or "").strip():
            yield block
    for table in page.get("tables", []):
        for cell in table.get("cells", []):
            if str(cell.get("source_text") or "").strip():
                yield cell


def page_input_fingerprint(page: dict[str, Any]) -> str:
    units = [
        {
            "unit_id": unit["unit_id"],
            "source_sha256": unit["source_sha256"],
            "bbox": unit["bbox"],
            "role": unit["role"],
        }
        for unit in iter_source_units(page)
    ]
    return fingerprint(
        {
            "document_sha256": page["document_sha256"],
            "page_number": page["page_number"],
            "media_box": page["media_box"],
            "rotation": page["rotation"],
            "units": units,
        }
    )


def extract_page(
    page: fitz.Page,
    *,
    document_sha256: str,
    extract_tables: bool,
) -> dict[str, Any]:
    raw = page.get_text("dict", sort=True)
    tables, warnings = _extract_tables(page) if extract_tables else ([], [])
    table_boxes = [table["bbox"] for table in tables]
    blocks: list[dict[str, Any]] = []
    for candidate in raw.get("blocks", []):
        if int(candidate.get("type", 0)) != 0:
            continue
        source_text, spans = _block_text(candidate)
        if not source_text:
            continue
        bbox = _round_rect(candidate.get("bbox", (0, 0, 0, 0)))
        if any(_overlap_ratio(bbox, table_bbox) >= 0.5 for table_bbox in table_boxes):
            continue
        unit_id = f"p{page.number + 1:04d}-b{len(blocks) + 1:04d}"
        blocks.append(
            {
                "unit_id": unit_id,
                "bbox": bbox,
                "source_text": source_text,
                "source_sha256": sha256_text(source_text),
                "role": "text",
                "spans": spans,
            }
        )
    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["unit_id"]))
    for index, block in enumerate(blocks, start=1):
        block["unit_id"] = f"p{page.number + 1:04d}-b{index:04d}"

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "extractor_revision": EXTRACTOR_REVISION,
        "document_sha256": document_sha256,
        "page_index": page.number,
        "page_number": page.number + 1,
        "page_label": page.get_label() or str(page.number + 1),
        "media_box": _round_rect(page.rect),
        "rotation": int(page.rotation),
        "blocks": blocks,
        "tables": tables,
        "warnings": warnings,
        "security_audit": _security_audit(blocks),
    }
    payload["input_fingerprint"] = page_input_fingerprint(payload)
    payload["unit_count"] = sum(1 for _ in iter_source_units(payload))
    payload["source_character_count"] = sum(
        len(str(unit["source_text"])) for unit in iter_source_units(payload)
    )
    if not payload["unit_count"] and page.get_images(full=True):
        payload["warnings"].append("image_only_page_ocr_may_be_required")
    return payload


def _checkpoint_is_current(path: Path, document_sha256: str) -> bool:
    try:
        payload = load_json(path)
        return (
            payload.get("extractor_revision") == EXTRACTOR_REVISION
            and payload.get("document_sha256") == document_sha256
            and payload.get("input_fingerprint") == page_input_fingerprint(payload)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def rebuild_source_manifests(
    profile: Profile,
    *,
    document_sha256: str,
    total_pages: int,
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for path in sorted(profile.extracted_dir.glob("page_*.json")):
        match = PAGE_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        page = load_json(path)
        if page.get("document_sha256") != document_sha256:
            continue
        pages.append(
            {
                "page_number": int(page["page_number"]),
                "input_fingerprint": page["input_fingerprint"],
                "unit_count": int(page["unit_count"]),
                "source_character_count": int(page["source_character_count"]),
                "security_audit_status": page["security_audit"]["status"],
                "warnings": list(page.get("warnings", [])),
            }
        )
        for unit in iter_source_units(page):
            units.append(
                {
                    "page_number": int(page["page_number"]),
                    "unit_id": unit["unit_id"],
                    "role": unit["role"],
                    "source_text": unit["source_text"],
                    "source_sha256": unit["source_sha256"],
                    "bbox": unit["bbox"],
                }
            )
    observed = {int(page["page_number"]) for page in pages}
    expected = set(range(1, total_pages + 1))
    summary = {
        "schema_version": "1.0",
        "extractor_revision": EXTRACTOR_REVISION,
        "document_sha256": document_sha256,
        "total_pages": total_pages,
        "extracted_pages": len(pages),
        "missing_pages": sorted(expected - observed),
        "unit_count": len(units),
        "source_character_count": sum(len(str(unit["source_text"])) for unit in units),
        "status": "complete" if observed == expected else "partial",
        "pages": sorted(pages, key=lambda item: int(item["page_number"])),
    }
    atomic_write_json(profile.manifests_dir / "source_pages.json", summary)
    atomic_write_jsonl(profile.manifests_dir / "source_units.jsonl", units)
    return summary


def extract_document(
    profile: Profile,
    *,
    pages: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if not profile.source_pdf.is_file():
        raise ExtractionError(f"Source PDF not found: {profile.source_pdf}")
    document_sha256 = sha256_file(profile.source_pdf)
    if profile.expected_sha256 and document_sha256 != profile.expected_sha256:
        raise ExtractionError(
            "Source SHA-256 mismatch: "
            f"expected {profile.expected_sha256}, got {document_sha256}"
        )
    with fitz.open(profile.source_pdf) as document:
        total_pages = document.page_count
        if profile.expected_pages and total_pages != profile.expected_pages:
            raise ExtractionError(
                f"Source page count mismatch: expected {profile.expected_pages}, got {total_pages}"
            )
        selected = parse_pages(pages, total_pages)
        profile.extracted_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        resumed = 0
        for page_number in selected:
            output = profile.extracted_dir / f"page_{page_number:04d}.json"
            if not force and output.is_file() and _checkpoint_is_current(
                output, document_sha256
            ):
                resumed += 1
                continue
            payload = extract_page(
                document[page_number - 1],
                document_sha256=document_sha256,
                extract_tables=profile.extract_tables,
            )
            atomic_write_json(output, payload)
            written += 1
    manifest = rebuild_source_manifests(
        profile,
        document_sha256=document_sha256,
        total_pages=total_pages,
    )
    return {
        "status": manifest["status"],
        "selected_pages": len(selected),
        "written_pages": written,
        "resumed_pages": resumed,
        "manifest": str(profile.manifests_dir / "source_pages.json"),
    }
