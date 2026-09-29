"""Prompt contracts for profile-relative scoring and synthesis."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from freshlit.nodes.filtering import (
    LLMEvaluation,
    LLMEvaluationBatch,
    ScoredPaper,
    _score_chunk,
    _score_paper,
)
from freshlit.nodes.ingestion import RawPaper
from freshlit.nodes.synthesis import (
    PaperExtraction,
    PaperExtractionBatch,
    _summarize_chunk,
    summarize_paper,
)


def _settings(profile: str):
    return SimpleNamespace(
        research_profile_text=profile,
        filtering=SimpleNamespace(llm_batch_size=10),
        llm=SimpleNamespace(workers=1),
    )


def _paper(paper_id: str, title: str, abstract: str) -> RawPaper:
    return RawPaper(
        id=paper_id,
        title=title,
        authors=["Taylor, Morgan"],
        publication_date="2026-09-01",
        venue_name="Cross-Disciplinary Review",
        abstract=abstract,
        source_type="journal",
    )


def _scored(paper: RawPaper) -> ScoredPaper:
    return ScoredPaper(
        raw_paper=paper,
        llm_eval=LLMEvaluation(
            relevance_score=8,
            passes_rubric=True,
            methodology_tags=["profile method"],
            fit_rationale="Aligned with the supplied profile.",
        ),
        journal_weight=1.0,
        final_score=8.0,
    )


class GenericScoringPromptTests(unittest.TestCase):
    def _assert_generic_rubric(self, prompt: str, profile: str) -> None:
        self.assertIn(profile, prompt)
        for score_band in ("9-10", "7-8", "4-6", "1-3"):
            self.assertIn(score_band, prompt)
        without_profile = prompt.replace(profile, "")
        for hardcoded in (
            "clonal",
            "somatic",
            "mutational burden",
            "mathematical/computational biologist",
            "purely clinical",
            "purely experimental",
        ):
            self.assertNotIn(hardcoded, without_profile.lower())
        self.assertIn("exclusions explicitly stated", without_profile)
        self.assertIn("untrusted paper content", without_profile.lower())

    @patch("freshlit.nodes.filtering.chat_structured")
    def test_single_scoring_uses_experimental_researcher_profile(self, chatMock):
        profile = (
            "# Research focus\nExperimental condensed-matter research on thin-film "
            "fabrication and transport measurements."
        )
        settings = _settings(profile)
        paper = _paper(
            "RAW-EXPERIMENT",
            "Thin-film transport under pressure",
            "The study fabricates samples and measures resistance across pressures.",
        )
        captured = {}

        def respond(client, supplied_settings, response_model, messages):
            captured["messages"] = messages
            return response_model(
                relevance_score=9,
                passes_rubric=True,
                methodology_tags=["transport measurement"],
                fit_rationale="Direct experimental fit.",
            )

        chatMock.side_effect = respond
        result = _score_paper(object(), settings, paper)

        self.assertEqual(result.relevance_score, 9)
        self._assert_generic_rubric(captured["messages"][0]["content"], profile)
        self.assertIn(
            "UNTRUSTED_PAPER_CONTENT_JSON", captured["messages"][1]["content"]
        )

    @patch("freshlit.nodes.filtering.chat_structured")
    def test_batch_scoring_uses_unrelated_humanities_profile(self, chatMock):
        profile = (
            "# Research focus\nMedieval manuscript circulation, scribal attribution, "
            "and archival provenance."
        )
        settings = _settings(profile)
        papers = [
            _paper("SECRET-1", "A manuscript catalogue", "Archival comparison."),
            _paper("SECRET-2", "Scribal hands", "Paleographic attribution."),
        ]
        captured = {}

        def respond(client, supplied_settings, response_model, messages):
            captured["messages"] = messages
            self.assertIs(response_model, LLMEvaluationBatch)
            return response_model.model_validate(
                {
                    "papers": [
                        {
                            "paper_id": f"P{index:04d}",
                            "relevance_score": 8,
                            "passes_rubric": True,
                            "methodology_tags": ["archival"],
                            "fit_rationale": "Profile fit.",
                        }
                        for index in (1, 2)
                    ]
                }
            )

        chatMock.side_effect = respond
        results = _score_chunk(object(), settings, papers)

        self.assertEqual(len(results), 2)
        self._assert_generic_rubric(captured["messages"][0]["content"], profile)
        user_prompt = captured["messages"][1]["content"]
        self.assertNotIn("SECRET-1", user_prompt)
        self.assertNotIn("SECRET-2", user_prompt)


class GenericSynthesisPromptTests(unittest.TestCase):
    def _assert_generic_extraction(self, prompt: str, profile: str) -> None:
        self.assertIn(profile, prompt)
        without_profile = prompt.replace(profile, "")
        self.assertIn("actual method used, regardless of discipline", without_profile)
        self.assertIn("quantitative results only when", without_profile)
        self.assertIn("any repository or archive", without_profile)
        self.assertIn("untrusted paper", without_profile.lower())
        for hardcoded in (
            "clonal",
            "somatic",
            "mathematical/computational biologist",
            "github",
            "purely clinical",
            "purely experimental",
        ):
            self.assertNotIn(hardcoded, without_profile.lower())

    @patch("freshlit.nodes.synthesis.chat_structured")
    def test_single_synthesis_includes_experimental_profile(self, chatMock):
        profile = (
            "# Research focus\nExperimental materials chemistry using operando "
            "spectroscopy and catalyst synthesis."
        )
        settings = _settings(profile)
        scored = _scored(
            _paper(
                "RAW-MATERIALS",
                "Operando catalyst characterization",
                "Spectroscopy measured conversion at several temperatures.",
            )
        )
        captured = {}

        def respond(client, supplied_settings, response_model, messages):
            captured["messages"] = messages
            return response_model(
                core_question="How does the catalyst change?",
                framework_and_method="Operando spectroscopy",
                key_finding="Conversion increased to 81%.",
                code_data_link="https://data.example/repository/42",
            )

        chatMock.side_effect = respond
        result = summarize_paper(object(), settings, scored)

        self.assertEqual(result.paper_id, "RAW-MATERIALS")
        self._assert_generic_extraction(captured["messages"][0]["content"], profile)
        self.assertIn(
            "UNTRUSTED_PAPER_CONTENT_JSON", captured["messages"][1]["content"]
        )

    @patch("freshlit.nodes.synthesis.chat_structured")
    def test_batch_synthesis_includes_environmental_profile(self, chatMock):
        profile = (
            "# Research focus\nUrban heat mapping with community sensors and "
            "geospatial field observations."
        )
        settings = _settings(profile)
        scored = [
            _scored(_paper("SECRET-A", "Urban heat", "Sensor observations.")),
            _scored(_paper("SECRET-B", "Tree canopy", "Geospatial survey.")),
        ]
        captured = {}

        def respond(client, supplied_settings, response_model, messages):
            captured["messages"] = messages
            self.assertIs(response_model, PaperExtractionBatch)
            return response_model.model_validate(
                {
                    "papers": [
                        {
                            "paper_id": f"P{index:04d}",
                            "core_question": "What shapes urban temperature?",
                            "framework_and_method": "Field sensor mapping",
                            "key_finding": "A spatial pattern was reported.",
                            "code_data_link": "None stated",
                        }
                        for index in (1, 2)
                    ]
                }
            )

        chatMock.side_effect = respond
        results = _summarize_chunk(object(), settings, scored)

        self.assertEqual([item.paper_id for item in results], ["SECRET-A", "SECRET-B"])
        self._assert_generic_extraction(captured["messages"][0]["content"], profile)
        user_prompt = captured["messages"][1]["content"]
        self.assertNotIn("SECRET-A", user_prompt)
        self.assertNotIn("SECRET-B", user_prompt)


if __name__ == "__main__":
    unittest.main()
