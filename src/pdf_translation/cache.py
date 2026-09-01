from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .io import canonical_json, sha256_text


CACHE_SCHEMA = "accepted-translations-v1"


class TranslationCache:
    """Process-safe SQLite cache that stores accepted candidates only."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS accepted_translation (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def key(source_text: str, role: str, pipeline_signature: str) -> str:
        return sha256_text(
            canonical_json(
                {
                    "schema": CACHE_SCHEMA,
                    "source_text": source_text,
                    "role": role,
                    "pipeline_signature": pipeline_signature,
                }
            )
        )

    def get(
        self, source_text: str, role: str, pipeline_signature: str
    ) -> dict[str, Any] | None:
        key = self.key(source_text, role, pipeline_signature)
        row = self.connection.execute(
            "SELECT payload FROM accepted_translation WHERE cache_key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def put(
        self,
        source_text: str,
        role: str,
        pipeline_signature: str,
        payload: dict[str, Any],
    ) -> None:
        if payload.get("status") != "accepted":
            raise ValueError("Only accepted translations may enter the cache")
        key = self.key(source_text, role, pipeline_signature)
        self.connection.execute(
            """
            INSERT INTO accepted_translation(cache_key, payload)
            VALUES (?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET payload = excluded.payload
            """,
            (key, canonical_json(payload)),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TranslationCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
