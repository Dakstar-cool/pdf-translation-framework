from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any, Iterable

from .backend import CascadeTranslator
from .cache import TranslationCache
from .config import Profile
from .extractor import iter_source_units
from .io import (
    atomic_write_json,
    atomic_write_jsonl,
    fingerprint,
    load_json,
    sha256_file,
)
from .validation import load_glossary, validate_candidate


PLAN_SCHEMA = "translation-shard-plan-v1"
TRANSLATION_SCHEMA = "translated-page-v1"
MERGE_SCHEMA = "translation-merge-v1"
QA_SCHEMA = "translation-qa-v1"
RELEASE_SCHEMA = "translation-release-v1"


class PipelineError(RuntimeError):
    """A pipeline stage was invoked with incomplete or inconsistent inputs."""


def _source_manifest(profile: Profile) -> dict[str, Any]:
    path = profile.manifests_dir / "source_pages.json"
    if not path.is_file():
        raise PipelineError("Source manifest is missing; run extract first")
    return load_json(path)


def _plan_path(profile: Profile) -> Path:
    return profile.translation_dir / "shard_plan.json"


def create_shard_plan(profile: Profile, shard_count: int) -> dict[str, Any]:
    if shard_count < 1:
        raise PipelineError("Shard count must be positive")
    source_path = profile.manifests_dir / "source_pages.json"
    source = _source_manifest(profile)
    pages = list(source.get("pages", []))
    if not pages:
        raise PipelineError("No extracted pages are available")
    actual_count = min(shard_count, len(pages))
    shards = [
        {"shard": index, "pages": [], "source_character_count": 0}
        for index in range(1, actual_count + 1)
    ]
    for page in sorted(
        pages,
        key=lambda item: (
            -int(item.get("source_character_count", 0)),
            int(item["page_number"]),
        ),
    ):
        target = min(
            shards,
            key=lambda item: (
                int(item["source_character_count"]),
                len(item["pages"]),
                int(item["shard"]),
            ),
        )
        target["pages"].append(int(page["page_number"]))
        target["source_character_count"] += int(
            page.get("source_character_count", 0)
        )
    for shard in shards:
        shard["pages"].sort()
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "profile_signature": profile.signature,
        "source_manifest_sha256": sha256_file(source_path),
        "document_sha256": source["document_sha256"],
        "source_status": source["status"],
        "requested_shards": shard_count,
        "shard_count": actual_count,
        "page_count": len(pages),
        "shards": shards,
    }
    plan["plan_signature"] = fingerprint(plan)
    atomic_write_json(_plan_path(profile), plan)
    return plan


def load_current_plan(profile: Profile) -> dict[str, Any]:
    path = _plan_path(profile)
    if not path.is_file():
        raise PipelineError("Shard plan is missing; run plan first")
    plan = load_json(path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise PipelineError("Unsupported shard-plan schema")
    if plan.get("profile_signature") != profile.signature:
        raise PipelineError("Profile changed after planning; create a new shard plan")
    source_path = profile.manifests_dir / "source_pages.json"
    if not source_path.is_file() or plan.get("source_manifest_sha256") != sha256_file(
        source_path
    ):
        raise PipelineError("Source manifest changed after planning; create a new shard plan")
    unsigned = dict(plan)
    declared = unsigned.pop("plan_signature", None)
    if declared != fingerprint(unsigned):
        raise PipelineError("Shard plan signature is invalid")
    return plan


def _translated_page_is_current(
    path: Path,
    *,
    source_page: dict[str, Any],
    plan_signature: str,
    pipeline_signature: str,
) -> bool:
    try:
        value = load_json(path)
        return (
            value.get("schema") == TRANSLATION_SCHEMA
            and value.get("input_fingerprint") == source_page["input_fingerprint"]
            and value.get("plan_signature") == plan_signature
            and value.get("pipeline_signature") == pipeline_signature
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _translate_page(
    source_page: dict[str, Any],
    *,
    translator: CascadeTranslator,
    shard_number: int,
    plan_signature: str,
) -> dict[str, Any]:
    translated_units: list[dict[str, Any]] = []
    for source_unit in iter_source_units(source_page):
        result = translator.translate(
            str(source_unit["source_text"]), str(source_unit["role"])
        )
        translated_units.append(
            {
                "unit_id": source_unit["unit_id"],
                "role": source_unit["role"],
                "bbox": source_unit["bbox"],
                "source_text": source_unit["source_text"],
                "source_sha256": source_unit["source_sha256"],
                **result,
            }
        )
    rejected = sum(
        1 for unit in translated_units if unit.get("status") != "accepted"
    )
    return {
        "schema": TRANSLATION_SCHEMA,
        "document_sha256": source_page["document_sha256"],
        "page_index": source_page["page_index"],
        "page_number": source_page["page_number"],
        "input_fingerprint": source_page["input_fingerprint"],
        "plan_signature": plan_signature,
        "pipeline_signature": translator.pipeline_signature,
        "shard": shard_number,
        "status": "accepted" if rejected == 0 else "rejected",
        "unit_count": len(translated_units),
        "accepted_unit_count": len(translated_units) - rejected,
        "rejected_unit_count": rejected,
        "units": translated_units,
    }


def translate_shard(
    profile: Profile,
    shard_number: int,
    *,
    force: bool = False,
) -> dict[str, Any]:
    plan = load_current_plan(profile)
    selected = next(
        (
            shard
            for shard in plan["shards"]
            if int(shard["shard"]) == shard_number
        ),
        None,
    )
    if selected is None:
        raise PipelineError(
            f"Shard {shard_number} does not exist in the current plan"
        )
    output_dir = (
        profile.translation_dir / "shards" / f"shard_{shard_number:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    translated = 0
    resumed = 0
    rejected = 0
    with TranslationCache(profile.cache_db) as cache:
        translator = CascadeTranslator(profile, cache)
        for page_number in selected["pages"]:
            source_path = profile.extracted_dir / f"page_{int(page_number):04d}.json"
            if not source_path.is_file():
                raise PipelineError(f"Extracted page is missing: {source_path}")
            source_page = load_json(source_path)
            output_path = output_dir / source_path.name
            if not force and output_path.is_file() and _translated_page_is_current(
                output_path,
                source_page=source_page,
                plan_signature=plan["plan_signature"],
                pipeline_signature=translator.pipeline_signature,
            ):
                resumed += 1
                existing = load_json(output_path)
                rejected += int(existing.get("status") != "accepted")
                continue
            output = _translate_page(
                source_page,
                translator=translator,
                shard_number=shard_number,
                plan_signature=plan["plan_signature"],
            )
            atomic_write_json(output_path, output)
            translated += 1
            rejected += int(output["status"] != "accepted")
    summary = {
        "shard": shard_number,
        "page_count": len(selected["pages"]),
        "translated_pages": translated,
        "resumed_pages": resumed,
        "rejected_pages": rejected,
        "status": "accepted" if rejected == 0 else "rejected",
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def translate_all_shards(
    profile: Profile,
    *,
    workers: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    plan = load_current_plan(profile)
    shard_numbers = [int(item["shard"]) for item in plan["shards"]]
    max_workers = max(1, min(int(workers), len(shard_numbers)))
    summaries: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                translate_shard, profile, shard, force=force
            ): shard
            for shard in shard_numbers
        }
        for future in concurrent.futures.as_completed(futures):
            summaries.append(future.result())
    summaries.sort(key=lambda item: int(item["shard"]))
    return {
        "status": (
            "accepted"
            if all(item["status"] == "accepted" for item in summaries)
            else "rejected"
        ),
        "workers": max_workers,
        "shards": summaries,
    }


def merge_shards(profile: Profile) -> dict[str, Any]:
    plan = load_current_plan(profile)
    merged_dir = profile.translation_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    page_entries: list[dict[str, Any]] = []
    accepted_pages = 0
    pipeline_signatures: set[str] = set()
    for shard in plan["shards"]:
        shard_number = int(shard["shard"])
        shard_dir = (
            profile.translation_dir / "shards" / f"shard_{shard_number:03d}"
        )
        for page_number in shard["pages"]:
            source_page = load_json(
                profile.extracted_dir / f"page_{int(page_number):04d}.json"
            )
            candidate_path = shard_dir / f"page_{int(page_number):04d}.json"
            if not candidate_path.is_file():
                issues.append(
                    {
                        "code": "missing_shard_page",
                        "shard": shard_number,
                        "page_number": int(page_number),
                    }
                )
                continue
            candidate = load_json(candidate_path)
            if (
                candidate.get("schema") != TRANSLATION_SCHEMA
                or candidate.get("page_number") != int(page_number)
                or candidate.get("shard") != shard_number
                or candidate.get("plan_signature") != plan["plan_signature"]
                or candidate.get("input_fingerprint")
                != source_page["input_fingerprint"]
            ):
                issues.append(
                    {
                        "code": "stale_or_invalid_shard_page",
                        "shard": shard_number,
                        "page_number": int(page_number),
                    }
                )
                continue
            output = merged_dir / candidate_path.name
            atomic_write_json(output, candidate)
            page_entries.append(
                {
                    "page_number": int(page_number),
                    "path": output.name,
                    "sha256": sha256_file(output),
                    "status": candidate["status"],
                }
            )
            accepted_pages += int(candidate.get("status") == "accepted")
            pipeline_signatures.add(str(candidate.get("pipeline_signature") or ""))
    manifest: dict[str, Any] = {
        "schema": MERGE_SCHEMA,
        "profile_signature": profile.signature,
        "plan_signature": plan["plan_signature"],
        "document_sha256": plan["document_sha256"],
        "expected_page_count": int(plan["page_count"]),
        "merged_page_count": len(page_entries),
        "accepted_page_count": accepted_pages,
        "pipeline_signatures": sorted(pipeline_signatures),
        "issues": issues,
        "pages": sorted(page_entries, key=lambda item: int(item["page_number"])),
    }
    manifest["status"] = (
        "complete"
        if not issues and len(page_entries) == int(plan["page_count"])
        else "incomplete"
    )
    manifest["merge_signature"] = fingerprint(manifest)
    atomic_write_json(merged_dir / "merge_manifest.json", manifest)
    return manifest


def _issue(
    issues: list[dict[str, Any]],
    code: str,
    *,
    page_number: int | None = None,
    unit_id: str | None = None,
    detail: Any = None,
) -> None:
    value: dict[str, Any] = {"code": code}
    if page_number is not None:
        value["page_number"] = page_number
    if unit_id is not None:
        value["unit_id"] = unit_id
    if detail is not None:
        value["detail"] = detail
    issues.append(value)


def validate_merged(profile: Profile) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    source = _source_manifest(profile)
    if source.get("status") != "complete":
        _issue(
            issues,
            "source_manifest_incomplete",
            detail={"missing_pages": source.get("missing_pages", [])},
        )
    if profile.expected_pages and int(source["total_pages"]) != profile.expected_pages:
        _issue(issues, "source_page_count_mismatch")
    if profile.expected_sha256 and source["document_sha256"] != profile.expected_sha256:
        _issue(issues, "source_sha256_mismatch")

    merge_path = profile.translation_dir / "merged" / "merge_manifest.json"
    if not merge_path.is_file():
        _issue(issues, "merge_manifest_missing")
        merge: dict[str, Any] = {"pages": []}
    else:
        merge = load_json(merge_path)
        if merge.get("schema") != MERGE_SCHEMA:
            _issue(issues, "merge_schema_invalid")
        if merge.get("profile_signature") != profile.signature:
            _issue(issues, "merge_profile_signature_mismatch")
        unsigned = dict(merge)
        declared = unsigned.pop("merge_signature", None)
        if declared != fingerprint(unsigned):
            _issue(issues, "merge_signature_invalid")
        if merge.get("status") != "complete":
            _issue(issues, "merge_incomplete", detail=merge.get("issues", []))

    merged_entries = {
        int(item["page_number"]): item for item in merge.get("pages", [])
    }
    glossary = load_glossary(profile.glossary_path)
    pipeline_signatures: set[str] = set()
    source_unit_count = 0
    accepted_unit_count = 0
    for page_summary in source.get("pages", []):
        page_number = int(page_summary["page_number"])
        if page_summary.get("security_audit_status") != "pass":
            _issue(issues, "source_security_warning", page_number=page_number)
        if "image_only_page_ocr_may_be_required" in page_summary.get("warnings", []):
            _issue(issues, "source_page_requires_ocr_review", page_number=page_number)
        source_page = load_json(
            profile.extracted_dir / f"page_{page_number:04d}.json"
        )
        source_units = {
            str(unit["unit_id"]): unit for unit in iter_source_units(source_page)
        }
        source_unit_count += len(source_units)
        entry = merged_entries.get(page_number)
        if entry is None:
            _issue(issues, "merged_page_missing", page_number=page_number)
            continue
        translated_path = (
            profile.translation_dir / "merged" / str(entry["path"])
        )
        if not translated_path.is_file():
            _issue(issues, "merged_page_file_missing", page_number=page_number)
            continue
        if sha256_file(translated_path) != entry.get("sha256"):
            _issue(issues, "merged_page_sha256_mismatch", page_number=page_number)
            continue
        translated_page = load_json(translated_path)
        if (
            translated_page.get("document_sha256") != source["document_sha256"]
            or translated_page.get("input_fingerprint")
            != source_page["input_fingerprint"]
        ):
            _issue(issues, "merged_page_source_mismatch", page_number=page_number)
            continue
        pipeline_signatures.add(
            str(translated_page.get("pipeline_signature") or "")
        )
        target_units_list = list(translated_page.get("units", []))
        target_units = {
            str(unit.get("unit_id")): unit for unit in target_units_list
        }
        if len(target_units) != len(target_units_list):
            _issue(issues, "duplicate_target_unit", page_number=page_number)
        missing_units = sorted(set(source_units) - set(target_units))
        extra_units = sorted(set(target_units) - set(source_units))
        if missing_units:
            _issue(
                issues,
                "target_units_missing",
                page_number=page_number,
                detail=missing_units,
            )
        if extra_units:
            _issue(
                issues,
                "target_units_extra",
                page_number=page_number,
                detail=extra_units,
            )
        for unit_id, source_unit in source_units.items():
            target_unit = target_units.get(unit_id)
            if target_unit is None:
                continue
            if target_unit.get("source_sha256") != source_unit["source_sha256"]:
                _issue(
                    issues,
                    "unit_source_sha256_mismatch",
                    page_number=page_number,
                    unit_id=unit_id,
                )
                continue
            if target_unit.get("status") != "accepted":
                _issue(
                    issues,
                    "unit_not_accepted",
                    page_number=page_number,
                    unit_id=unit_id,
                    detail=target_unit.get("attempts"),
                )
                continue
            candidate_issues = validate_candidate(
                str(source_unit["source_text"]),
                str(target_unit.get("translation") or ""),
                profile.validation,
                glossary,
            )
            if candidate_issues:
                _issue(
                    issues,
                    "stored_candidate_revalidation_failed",
                    page_number=page_number,
                    unit_id=unit_id,
                    detail=candidate_issues,
                )
                continue
            accepted_unit_count += 1
    if len(pipeline_signatures) != 1 or not next(iter(pipeline_signatures), ""):
        _issue(
            issues,
            "pipeline_signature_set_invalid",
            detail=sorted(pipeline_signatures),
        )

    summary: dict[str, Any] = {
        "schema": QA_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "profile_signature": profile.signature,
        "document_sha256": source.get("document_sha256"),
        "source_page_count": len(source.get("pages", [])),
        "merged_page_count": len(merged_entries),
        "source_unit_count": source_unit_count,
        "accepted_unit_count": accepted_unit_count,
        "issue_count": len(issues),
        "pipeline_signatures": sorted(pipeline_signatures),
    }
    summary["qa_signature"] = fingerprint(summary)
    atomic_write_json(profile.qa_dir / "summary.json", summary)
    atomic_write_jsonl(profile.qa_dir / "issues.jsonl", issues)
    return {**summary, "issues": issues}


def _release_rows(pages: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for page in pages:
        for unit in page.get("units", []):
            yield {
                "page_number": page["page_number"],
                "unit_id": unit["unit_id"],
                "role": unit["role"],
                "bbox": unit["bbox"],
                "source_text": unit["source_text"],
                "source_sha256": unit["source_sha256"],
                "translation": unit["translation"],
                "backend": unit["backend"],
                "pipeline_signature": unit["pipeline_signature"],
            }


def create_release(profile: Profile) -> dict[str, Any]:
    qa = validate_merged(profile)
    if qa["status"] != "PASS":
        raise PipelineError(
            f"Release blocked by {qa['issue_count']} QA issue(s); "
            f"see {profile.qa_dir / 'issues.jsonl'}"
        )
    merge = load_json(
        profile.translation_dir / "merged" / "merge_manifest.json"
    )
    pipeline_signature = str(qa["pipeline_signatures"][0])
    release_id = (
        f"{str(qa['document_sha256'])[:12].lower()}-"
        f"{pipeline_signature[:12].lower()}"
    )
    destination = profile.release_dir / release_id
    pages_dir = destination / "pages"
    pages: list[dict[str, Any]] = []
    page_entries: list[dict[str, Any]] = []
    for entry in sorted(merge["pages"], key=lambda item: int(item["page_number"])):
        page = load_json(
            profile.translation_dir / "merged" / str(entry["path"])
        )
        output = pages_dir / f"page_{int(page['page_number']):04d}.json"
        atomic_write_json(output, page)
        pages.append(page)
        page_entries.append(
            {
                "page_number": int(page["page_number"]),
                "path": f"pages/{output.name}",
                "sha256": sha256_file(output),
            }
        )
    catalog_path = destination / "translations.jsonl"
    atomic_write_jsonl(catalog_path, _release_rows(pages))
    manifest: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "release_id": release_id,
        "project_id": profile.project_id,
        "profile_signature": profile.signature,
        "document_sha256": qa["document_sha256"],
        "pipeline_signature": pipeline_signature,
        "qa_signature": qa["qa_signature"],
        "page_count": len(pages),
        "unit_count": qa["accepted_unit_count"],
        "translations_jsonl": catalog_path.name,
        "translations_sha256": sha256_file(catalog_path),
        "pages": page_entries,
    }
    manifest["release_signature"] = fingerprint(manifest)
    atomic_write_json(destination / "manifest.json", manifest)
    atomic_write_json(
        profile.release_dir / "current.json",
        {
            "release_id": release_id,
            "manifest": f"{release_id}/manifest.json",
            "release_signature": manifest["release_signature"],
        },
    )
    return manifest
