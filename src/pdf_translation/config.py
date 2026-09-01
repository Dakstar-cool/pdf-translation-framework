from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import fingerprint, sha256_file


class ProfileError(ValueError):
    """The selected translation profile is invalid."""


def _required_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileError(f"{label} must be a JSON object")
    return dict(value)


def _required_text(value: Any, label: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise ProfileError(f"{label} must not be empty")
    return rendered


def _resolve(base: Path, value: Any, label: str) -> Path:
    raw = _required_text(value, label)
    candidate = Path(raw).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


@dataclass(frozen=True)
class BackendConfig:
    kind: str
    endpoint: str
    model: str
    timeout_seconds: float = 300.0
    max_tokens: int = 2048
    temperature: float = 0.0
    seed: int = 1
    api_key_env: str | None = None

    @classmethod
    def from_mapping(cls, value: Any, label: str) -> "BackendConfig":
        raw = _required_mapping(value, label)
        kind = _required_text(raw.get("kind"), f"{label}.kind")
        if kind not in {"openai_chat", "llamacpp_completion", "ollama_chat"}:
            raise ProfileError(
                f"{label}.kind must be openai_chat, llamacpp_completion, or ollama_chat"
            )
        return cls(
            kind=kind,
            endpoint=_required_text(raw.get("endpoint"), f"{label}.endpoint"),
            model=_required_text(raw.get("model"), f"{label}.model"),
            timeout_seconds=max(1.0, float(raw.get("timeout_seconds", 300))),
            max_tokens=max(32, int(raw.get("max_tokens", 2048))),
            temperature=float(raw.get("temperature", 0.0)),
            seed=int(raw.get("seed", 1)),
            api_key_env=(
                _required_text(raw["api_key_env"], f"{label}.api_key_env")
                if raw.get("api_key_env")
                else None
            ),
        )


@dataclass(frozen=True)
class ValidationConfig:
    protect_numbers: bool
    protect_units: bool
    protect_intervals: bool
    negation_mode: str
    source_negation_patterns: tuple[str, ...]
    target_negation_patterns: tuple[str, ...]
    unit_pattern: str
    interval_pattern: str


@dataclass(frozen=True)
class Profile:
    path: Path
    raw: dict[str, Any]
    project_id: str
    source_pdf: Path
    workspace: Path
    release_dir: Path
    expected_sha256: str | None
    expected_pages: int | None
    extract_tables: bool
    source_language: str
    target_language: str
    prompt_template: str
    glossary_path: Path | None
    primary: BackendConfig
    fallback: BackendConfig | None
    validation: ValidationConfig

    @classmethod
    def load(cls, path: Path) -> "Profile":
        resolved = path.resolve()
        try:
            raw_value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"Cannot read profile {resolved}: {exc}") from exc
        raw = _required_mapping(raw_value, "profile")
        if str(raw.get("schema_version")) != "1.0":
            raise ProfileError("profile.schema_version must be '1.0'")
        base = resolved.parent
        project = _required_mapping(raw.get("project"), "project")
        document = _required_mapping(raw.get("document"), "document")
        translation = _required_mapping(raw.get("translation"), "translation")
        validation = _required_mapping(raw.get("validation"), "validation")

        expected_sha = str(document.get("expected_sha256") or "").strip().upper() or None
        if expected_sha is not None and (
            len(expected_sha) != 64
            or any(character not in "0123456789ABCDEF" for character in expected_sha)
        ):
            raise ProfileError("document.expected_sha256 must be null or 64 hex characters")
        expected_pages = document.get("expected_pages")
        if expected_pages is not None:
            expected_pages = int(expected_pages)
            if expected_pages < 1:
                raise ProfileError("document.expected_pages must be positive")

        prompt_template = _required_text(
            translation.get("prompt_template"), "translation.prompt_template"
        )
        for placeholder in ("{source_language}", "{target_language}"):
            if placeholder not in prompt_template:
                raise ProfileError(
                    f"translation.prompt_template must contain {placeholder}"
                )
        glossary_path = (
            _resolve(base, translation["glossary"], "translation.glossary")
            if translation.get("glossary")
            else None
        )
        if glossary_path is not None and not glossary_path.is_file():
            raise ProfileError(f"Glossary not found: {glossary_path}")

        negation_mode = str(validation.get("negation_mode", "count"))
        if negation_mode not in {"off", "presence", "count"}:
            raise ProfileError("validation.negation_mode must be off, presence, or count")
        source_patterns = tuple(
            _required_text(item, "validation.source_negation_patterns[]")
            for item in validation.get("source_negation_patterns", [])
        )
        target_patterns = tuple(
            _required_text(item, "validation.target_negation_patterns[]")
            for item in validation.get("target_negation_patterns", [])
        )
        if negation_mode != "off" and (not source_patterns or not target_patterns):
            raise ProfileError("Negation patterns are required when negation QA is enabled")

        return cls(
            path=resolved,
            raw=raw,
            project_id=_required_text(project.get("id"), "project.id"),
            source_pdf=_resolve(base, project.get("source_pdf"), "project.source_pdf"),
            workspace=_resolve(base, project.get("workspace"), "project.workspace"),
            release_dir=_resolve(base, project.get("release_dir"), "project.release_dir"),
            expected_sha256=expected_sha,
            expected_pages=expected_pages,
            extract_tables=bool(document.get("extract_tables", True)),
            source_language=_required_text(
                translation.get("source_language"), "translation.source_language"
            ),
            target_language=_required_text(
                translation.get("target_language"), "translation.target_language"
            ),
            prompt_template=prompt_template,
            glossary_path=glossary_path,
            primary=BackendConfig.from_mapping(translation.get("primary"), "translation.primary"),
            fallback=(
                BackendConfig.from_mapping(translation["fallback"], "translation.fallback")
                if translation.get("fallback")
                else None
            ),
            validation=ValidationConfig(
                protect_numbers=bool(validation.get("protect_numbers", True)),
                protect_units=bool(validation.get("protect_units", True)),
                protect_intervals=bool(validation.get("protect_intervals", True)),
                negation_mode=negation_mode,
                source_negation_patterns=source_patterns,
                target_negation_patterns=target_patterns,
                unit_pattern=_required_text(
                    validation.get("unit_pattern"), "validation.unit_pattern"
                ),
                interval_pattern=_required_text(
                    validation.get("interval_pattern"), "validation.interval_pattern"
                ),
            ),
        )

    @property
    def extracted_dir(self) -> Path:
        return self.workspace / "extracted"

    @property
    def manifests_dir(self) -> Path:
        return self.workspace / "manifests"

    @property
    def translation_dir(self) -> Path:
        return self.workspace / "translation"

    @property
    def qa_dir(self) -> Path:
        return self.workspace / "qa"

    @property
    def cache_db(self) -> Path:
        return self.translation_dir / "cache.sqlite3"

    @property
    def signature(self) -> str:
        payload: dict[str, Any] = {"profile": self.raw}
        if self.glossary_path is not None:
            payload["glossary_sha256"] = sha256_file(self.glossary_path)
        return fingerprint(payload)
