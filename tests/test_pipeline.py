from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from pdf_translation.config import Profile
from pdf_translation.extractor import extract_document
from pdf_translation.io import (
    atomic_write_json,
    fingerprint,
    load_json,
    sha256_file,
)
from pdf_translation.pipeline import (
    PipelineError,
    create_release,
    create_shard_plan,
    merge_shards,
    translate_all_shards,
    validate_merged,
)

from .helpers import write_profile


class FakeCascade:
    def __init__(self, profile: Profile, _cache: object):
        self.profile = profile
        self.pipeline_signature = fingerprint(
            {"profile": profile.signature, "fake": True}
        )

    def translate(self, source_text: str, _role: str) -> dict:
        return {
            "status": "accepted",
            "translation": source_text,
            "backend": "fake",
            "pipeline_signature": self.pipeline_signature,
            "attempts": [
                {
                    "backend": "fake",
                    "status": "accepted",
                    "issues": [],
                    "error": None,
                }
            ],
            "cache_hit": False,
        }


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "document.pdf"
        document = pymupdf.open()
        first = document.new_page()
        first.insert_text((72, 72), "Alpha 5 mg.")
        second = document.new_page()
        second.insert_text((72, 72), "Beta.")
        document.save(self.source)
        document.close()
        self.profile = Profile.load(
            write_profile(
                self.root,
                source_pdf=self.source,
                expected_pages=2,
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_checkpointed_release_flow(self) -> None:
        extraction = extract_document(self.profile)
        self.assertEqual(extraction["status"], "complete")
        resumed = extract_document(self.profile)
        self.assertEqual(resumed["resumed_pages"], 2)

        plan = create_shard_plan(self.profile, 2)
        self.assertEqual(plan["shard_count"], 2)
        with patch("pdf_translation.pipeline.CascadeTranslator", FakeCascade):
            translation = translate_all_shards(self.profile, workers=2)
        self.assertEqual(translation["status"], "accepted")

        merge = merge_shards(self.profile)
        self.assertEqual(merge["status"], "complete")
        qa = validate_merged(self.profile)
        self.assertEqual(qa["status"], "PASS", qa["issues"])

        release = create_release(self.profile)
        self.assertEqual(release["page_count"], 2)
        self.assertEqual(release["unit_count"], 2)
        self.assertTrue(
            (self.profile.release_dir / "current.json").is_file()
        )

        merged_dir = self.profile.translation_dir / "merged"
        page_path = merged_dir / "page_0001.json"
        page = load_json(page_path)
        page["units"][0]["translation"] = "Alpha 6 mg."
        atomic_write_json(page_path, page)
        merge_path = merged_dir / "merge_manifest.json"
        merge_manifest = load_json(merge_path)
        merge_manifest["pages"][0]["sha256"] = sha256_file(page_path)
        merge_manifest.pop("merge_signature")
        merge_manifest["merge_signature"] = fingerprint(merge_manifest)
        atomic_write_json(merge_path, merge_manifest)

        failed_qa = validate_merged(self.profile)
        self.assertEqual(failed_qa["status"], "FAIL")
        self.assertTrue(
            any(
                issue["code"] == "stored_candidate_revalidation_failed"
                for issue in failed_qa["issues"]
            )
        )
        with self.assertRaises(PipelineError):
            create_release(self.profile)


if __name__ == "__main__":
    unittest.main()
