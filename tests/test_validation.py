from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pdf_translation.backend import CascadeTranslator
from pdf_translation.cache import TranslationCache
from pdf_translation.config import Profile
from pdf_translation.validation import (
    GlossaryEntry,
    protect_text,
    restore_text,
    validate_candidate,
)

from .helpers import write_profile


class FakeBackend:
    def __init__(self, name: str, response: str):
        self.name = name
        self.response = response
        self.calls = 0

    def generate(self, _system_prompt: str, _user_prompt: str) -> str:
        self.calls += 1
        return self.response


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "source.pdf"
        source.write_bytes(b"not-used")
        self.profile = Profile.load(
            write_profile(
                self.root,
                source_pdf=source,
                negation_mode="count",
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_protection_round_trip_is_exact(self) -> None:
        protected = protect_text(
            "Give 2.5-5 mg q 8 h.", self.profile.validation
        )
        self.assertIn("__PTF_0001__", protected.masked)
        restored, issues = restore_text(protected.masked, protected)
        self.assertEqual(restored, "Give 2.5-5 mg q 8 h.")
        self.assertEqual(issues, [])

    def test_validator_blocks_numbers_negation_and_glossary(self) -> None:
        issues = validate_candidate(
            "Do not give 5 mg of active agent.",
            "Дать 6 mg активного вещества.",
            self.profile.validation,
            (GlossaryEntry("active agent", "действующее вещество"),),
        )
        codes = {issue["code"] for issue in issues}
        self.assertIn("number_signature_mismatch", codes)
        self.assertIn("negation_mismatch", codes)
        self.assertIn("glossary_target_missing", codes)

    def test_cascade_uses_fallback_and_caches_only_accepted_result(self) -> None:
        primary = FakeBackend("primary", "Неверный ответ")
        fallback = FakeBackend(
            "fallback", "Не использовать __PTF_0001__."
        )
        with TranslationCache(self.root / "cache.sqlite3") as cache:
            cascade = CascadeTranslator(
                self.profile,
                cache,
                primary=primary,
                fallback=fallback,
            )
            result = cascade.translate("Do not use 5 mg.", "text")
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(result["backend"], "fallback")
            self.assertEqual(result["translation"], "Не использовать 5 mg.")
            again = cascade.translate("Do not use 5 mg.", "text")
            self.assertTrue(again["cache_hit"])
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)


if __name__ == "__main__":
    unittest.main()
