from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .cache import TranslationCache
from .config import BackendConfig, Profile
from .io import fingerprint
from .validation import (
    VALIDATOR_REVISION,
    GlossaryEntry,
    load_glossary,
    matching_glossary,
    protect_text,
    restore_text,
    validate_candidate,
)


class BackendError(RuntimeError):
    """A local model endpoint failed or returned an invalid response."""


class TextBackend(Protocol):
    name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class HTTPBackend:
    MAX_RESPONSE_BYTES = 16 * 1024 * 1024

    def __init__(self, config: BackendConfig):
        self.config = config
        self.name = f"{config.kind}:{config.model}"

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        config = self.config
        if config.kind == "openai_chat":
            return {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "seed": config.seed,
                "stream": False,
            }
        if config.kind == "ollama_chat":
            return {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                    "seed": config.seed,
                },
                "stream": False,
            }
        return {
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "temperature": config.temperature,
            "n_predict": config.max_tokens,
            "seed": config.seed,
            "cache_prompt": True,
            "stream": False,
        }

    def _content(self, response: dict[str, Any]) -> str:
        if self.config.kind == "openai_chat":
            try:
                return str(response["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError) as exc:
                raise BackendError("Invalid OpenAI-compatible response") from exc
        if self.config.kind == "ollama_chat":
            try:
                return str(response["message"]["content"])
            except (KeyError, TypeError) as exc:
                raise BackendError("Invalid Ollama response") from exc
        content = response.get("content")
        if content is None:
            raise BackendError("Invalid llama.cpp completion response")
        return str(content)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            secret = os.environ.get(self.config.api_key_env)
            if not secret:
                raise BackendError(
                    f"Environment variable {self.config.api_key_env!r} is not set"
                )
            headers["Authorization"] = f"Bearer {secret}"
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(self._payload(system_prompt, user_prompt)).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                raw = response.read(self.MAX_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise BackendError(f"{self.name} request failed: {exc}") from exc
        if len(raw) > self.MAX_RESPONSE_BYTES:
            raise BackendError(f"{self.name} response exceeded the size limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(f"{self.name} returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise BackendError(f"{self.name} returned a non-object response")
        return self._content(payload).strip()


@dataclass(frozen=True)
class TranslationAttempt:
    backend: str
    status: str
    issues: tuple[dict[str, Any], ...] = ()
    error: str | None = None


class CascadeTranslator:
    def __init__(
        self,
        profile: Profile,
        cache: TranslationCache,
        *,
        primary: TextBackend | None = None,
        fallback: TextBackend | None = None,
    ):
        self.profile = profile
        self.cache = cache
        self.primary = primary or HTTPBackend(profile.primary)
        self.fallback = (
            fallback
            if fallback is not None
            else (HTTPBackend(profile.fallback) if profile.fallback else None)
        )
        self.glossary: tuple[GlossaryEntry, ...] = load_glossary(
            profile.glossary_path
        )
        self.pipeline_signature = fingerprint(
            {
                "profile_signature": profile.signature,
                "validator_revision": VALIDATOR_REVISION,
                "primary": asdict(profile.primary),
                "fallback": asdict(profile.fallback) if profile.fallback else None,
            }
        )

    def _prompts(self, source_text: str, role: str) -> tuple[str, str, Any]:
        protected = protect_text(source_text, self.profile.validation)
        system = self.profile.prompt_template.format(
            source_language=self.profile.source_language,
            target_language=self.profile.target_language,
        )
        terms = matching_glossary(source_text, self.glossary)
        glossary_instruction = ""
        if terms:
            rendered = "; ".join(f"{item.source} => {item.target}" for item in terms)
            glossary_instruction = f"\nRequired terminology: {rendered}"
        user = f"Role: {role}{glossary_instruction}\n\nText:\n{protected.masked}"
        return system, user, protected

    def translate(self, source_text: str, role: str) -> dict[str, Any]:
        cached = self.cache.get(source_text, role, self.pipeline_signature)
        if cached is not None:
            result = dict(cached)
            result["cache_hit"] = True
            return result

        system, user, protected = self._prompts(source_text, role)
        attempts: list[TranslationAttempt] = []
        backends = [self.primary, *([self.fallback] if self.fallback else [])]
        for backend in backends:
            assert backend is not None
            try:
                raw_candidate = backend.generate(system, user)
                candidate, marker_issues = restore_text(raw_candidate, protected)
                issues = [
                    *marker_issues,
                    *validate_candidate(
                        source_text,
                        candidate,
                        self.profile.validation,
                        self.glossary,
                    ),
                ]
                if not issues:
                    accepted = {
                        "status": "accepted",
                        "translation": candidate,
                        "backend": backend.name,
                        "pipeline_signature": self.pipeline_signature,
                        "attempts": [
                            asdict(item)
                            for item in [
                                *attempts,
                                TranslationAttempt(backend.name, "accepted"),
                            ]
                        ],
                        "cache_hit": False,
                    }
                    self.cache.put(
                        source_text, role, self.pipeline_signature, accepted
                    )
                    return accepted
                attempts.append(
                    TranslationAttempt(
                        backend=backend.name,
                        status="rejected",
                        issues=tuple(issues),
                    )
                )
            except Exception as exc:
                attempts.append(
                    TranslationAttempt(
                        backend=backend.name,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return {
            "status": "rejected",
            "translation": None,
            "backend": None,
            "pipeline_signature": self.pipeline_signature,
            "attempts": [asdict(item) for item in attempts],
            "cache_hit": False,
        }
