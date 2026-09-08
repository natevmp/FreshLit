"""Focused contracts for structured LLM responses and batch ID mapping."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from freshlit.nodes.filtering import (
    LLMEvaluation,
    LLMEvaluationBatch,
    _score_chunk,
    _score_paper,
)
from freshlit.nodes.ingestion import RawPaper
from freshlit.nodes.synthesis import (
    FieldPulse,
    PaperExtraction,
    PaperExtractionBatch,
    _summarize_chunk,
    generate_field_pulse,
    summarize_paper,
)
from freshlit.nodes.filtering import ScoredPaper
from freshlit.utils.llm import LLMNoFallbackError


def _settings():
    return SimpleNamespace(
        research_profile_text="Test profile",
        filtering=SimpleNamespace(llm_batch_size=10),
        llm=SimpleNamespace(workers=1),
    )


def _paper(rawId: str, title: str) -> RawPaper:
    return RawPaper(
        id=rawId,
        doi=None,
        title=title,
        authors=["A. Author"],
        publication_date="2026-09-01",
        venue_name="Test Journal",
        abstract="A sufficiently detailed abstract about clone growth and modelling.",
        source_type="journal",
    )


def _evaluation(score: int, rationale: str) -> LLMEvaluation:
    return LLMEvaluation(
        relevance_score=score,
        passes_rubric=score >= 7,
        methodology_tags=[rationale],
        fit_rationale=rationale,
    )


def _scored(rawId: str, title: str) -> ScoredPaper:
    return ScoredPaper(
        raw_paper=_paper(rawId, title),
        llm_eval=_evaluation(8, f"rationale for {title}"),
        journal_weight=1.0,
        final_score=8.0,
    )


class FilteringBatchContractTests(unittest.TestCase):
    @patch("freshlit.nodes.filtering.chat_structured")
    def test_out_of_order_scoring_is_mapped_to_input_order(self, chatMock):
        _paperid = [
            _paper("RAW-FIRST", "First title"),
            _paper("RAW-SECOND", "Second title"),
        ]

        def respond(client, settings, response_model, messages):
            self.assertIs(response_model, LLMEvaluationBatch)
            prompt = messages[-1]["content"]
            self.assertIn("P0001", prompt)
            self.assertIn("P0002", prompt)
            self.assertNotIn("RAW-FIRST", prompt)
            self.assertNotIn("RAW-SECOND", prompt)
            return response_model.model_validate(
                {
                    "papers": [
                        {
                            "paper_id": "P0002",
                            "relevance_score": 9,
                            "passes_rubric": True,
                            "methodology_tags": ["second"],
                            "fit_rationale": "second result",
                        },
                        {
                            "paper_id": "P0001",
                            "relevance_score": 7,
                            "passes_rubric": True,
                            "methodology_tags": ["first"],
                            "fit_rationale": "first result",
                        },
                    ]
                }
            )

        chatMock.side_effect = respond
        evaluation_paperid = _score_chunk(object(), _settings(), _paperid)

        self.assertEqual(
            [evaluation.fit_rationale for evaluation in evaluation_paperid],
            ["first result", "second result"],
        )
        self.assertTrue(all(type(item) is LLMEvaluation for item in evaluation_paperid))

    def test_bad_batch_ids_trigger_per_paper_fallback(self):
        _paperid = [_paper("RAW-A", "Paper A"), _paper("RAW-B", "Paper B")]
        malformed_paperid = {
            "duplicate": ["P0001", "P0001"],
            "missing": ["P0001"],
            "unknown": ["P0001", "P9999"],
        }

        for contractCase, responseIds in malformed_paperid.items():
            with self.subTest(contractCase=contractCase), patch(
                "freshlit.nodes.filtering.chat_structured"
            ) as chatMock:
                batch = LLMEvaluationBatch.model_validate(
                    {
                        "papers": [
                            {
                                "paper_id": responseId,
                                "relevance_score": 8,
                                "passes_rubric": True,
                                "methodology_tags": [responseId],
                                "fit_rationale": responseId,
                            }
                            for responseId in responseIds
                        ]
                    }
                )
                chatMock.side_effect = [
                    batch,
                    _evaluation(7, "fallback A"),
                    _evaluation(8, "fallback B"),
                ]

                evaluation_paperid = _score_chunk(object(), _settings(), _paperid)

                self.assertEqual(chatMock.call_count, 3)
                self.assertEqual(
                    [evaluation.fit_rationale for evaluation in evaluation_paperid],
                    ["fallback A", "fallback B"],
                )

    @patch("freshlit.nodes.filtering.chat_structured")
    def test_no_fallback_error_from_batch_propagates_without_fallback(self, chatMock):
        chatMock.side_effect = LLMNoFallbackError("request timed out")
        papers_paperid = [_paper("RAW-A", "Paper A"), _paper("RAW-B", "Paper B")]

        with self.assertRaisesRegex(LLMNoFallbackError, "timed out"):
            _score_chunk(object(), _settings(), papers_paperid)

        chatMock.assert_called_once()

    @patch("freshlit.nodes.filtering.chat_structured")
    def test_no_fallback_error_from_single_paper_propagates(self, chatMock):
        chatMock.side_effect = LLMNoFallbackError("unsafe activity")

        with self.assertRaisesRegex(LLMNoFallbackError, "unsafe activity"):
            _score_paper(object(), _settings(), _paper("RAW-A", "Paper A"))

        chatMock.assert_called_once()


class StructuredResponseValidationTests(unittest.TestCase):
    def test_evaluation_score_and_pass_flag_must_be_consistent(self):
        self.assertEqual(_evaluation(7, "valid").relevance_score, 7)

        for invalidScore in (0, 11, "7"):
            with self.subTest(invalidScore=invalidScore), self.assertRaises(
                ValidationError
            ):
                LLMEvaluation(
                    relevance_score=invalidScore,
                    passes_rubric=True,
                    methodology_tags=[],
                    fit_rationale="invalid",
                )

        for score, passes in ((6, True), (7, False)):
            with self.subTest(score=score, passes=passes), self.assertRaises(
                ValidationError
            ):
                LLMEvaluation(
                    relevance_score=score,
                    passes_rubric=passes,
                    methodology_tags=[],
                    fit_rationale="inconsistent",
                )

    def test_field_pulse_requires_exactly_three_trends(self):
        self.assertEqual(len(FieldPulse(trends=["one", "two", "three"]).trends), 3)
        for trend_pulseid in (["one", "two"], ["one", "two", "three", "four"]):
            with self.subTest(trends=trend_pulseid), self.assertRaises(ValidationError):
                FieldPulse(trends=trend_pulseid)

    def test_llm_response_models_reject_extra_fields(self):
        validEvaluation = {
            "relevance_score": 8,
            "passes_rubric": True,
            "methodology_tags": [],
            "fit_rationale": "fit",
        }
        validExtraction = {
            "core_question": "question",
            "framework_and_method": "method",
            "key_finding": "finding",
            "code_data_link": "None stated",
        }
        modelAndPayload = [
            (LLMEvaluation, validEvaluation),
            (
                LLMEvaluationBatch,
                {"papers": [{"paper_id": "P0001", **validEvaluation}]},
            ),
            (PaperExtraction, validExtraction),
            (
                PaperExtractionBatch,
                {"papers": [{"paper_id": "P0001", **validExtraction}]},
            ),
            (FieldPulse, {"trends": ["one", "two", "three"]}),
        ]

        for model, payload in modelAndPayload:
            with self.subTest(model=model.__name__), self.assertRaises(ValidationError):
                model.model_validate({**payload, "unexpected": "forbidden"})

        with self.assertRaises(ValidationError):
            PaperExtraction(
                core_question="question",
                framework_and_method="method",
                key_finding="finding",
            )


class SynthesisBatchContractTests(unittest.TestCase):
    @patch("freshlit.nodes.synthesis.chat_structured")
    def test_out_of_order_extractions_are_mapped_to_input_order(self, chatMock):
        scored_paperid = [
            _scored("RAW/SECRET/ONE", "First extraction title"),
            _scored("RAW/SECRET/TWO", "Second extraction title"),
        ]

        def respond(client, settings, response_model, messages):
            self.assertIs(response_model, PaperExtractionBatch)
            prompt = messages[-1]["content"]
            self.assertNotIn("RAW/SECRET/ONE", prompt)
            self.assertNotIn("RAW/SECRET/TWO", prompt)
            return response_model.model_validate(
                {
                    "papers": [
                        {
                            "paper_id": "P0002",
                            "core_question": "second question",
                            "framework_and_method": "second method",
                            "key_finding": "second finding",
                            "code_data_link": "second link",
                        },
                        {
                            "paper_id": "P0001",
                            "core_question": "first question",
                            "framework_and_method": "first method",
                            "key_finding": "first finding",
                            "code_data_link": "first link",
                        },
                    ]
                }
            )

        chatMock.side_effect = respond
        summary_paperid = _summarize_chunk(object(), _settings(), scored_paperid)

        self.assertEqual(
            [summary.paper_id for summary in summary_paperid],
            ["RAW/SECRET/ONE", "RAW/SECRET/TWO"],
        )
        self.assertEqual(
            [summary.core_question for summary in summary_paperid],
            ["first question", "second question"],
        )

    @patch("freshlit.nodes.synthesis.chat_structured")
    def test_bad_extraction_ids_trigger_per_paper_fallback(self, chatMock):
        scored_paperid = [
            _scored("RAW-A", "Paper A"),
            _scored("RAW-B", "Paper B"),
        ]
        chatMock.side_effect = [
            PaperExtractionBatch.model_validate(
                {
                    "papers": [
                        {
                            "paper_id": "P0001",
                            "core_question": "batch A",
                            "framework_and_method": "batch method",
                            "key_finding": "batch finding",
                            "code_data_link": "None stated",
                        },
                        {
                            "paper_id": "P9999",
                            "core_question": "unknown",
                            "framework_and_method": "batch method",
                            "key_finding": "batch finding",
                            "code_data_link": "None stated",
                        },
                    ]
                }
            ),
            PaperExtraction(
                core_question="fallback A",
                framework_and_method="method A",
                key_finding="finding A",
                code_data_link="None stated",
            ),
            PaperExtraction(
                core_question="fallback B",
                framework_and_method="method B",
                key_finding="finding B",
                code_data_link="None stated",
            ),
        ]

        summary_paperid = _summarize_chunk(object(), _settings(), scored_paperid)

        self.assertEqual(chatMock.call_count, 3)
        self.assertEqual(
            [summary.core_question for summary in summary_paperid],
            ["fallback A", "fallback B"],
        )

    @patch("freshlit.nodes.synthesis.chat_structured")
    def test_no_fallback_error_from_batch_propagates_without_fallback(self, chatMock):
        chatMock.side_effect = LLMNoFallbackError("unsafe activity")
        scored_paperid = [_scored("RAW-A", "Paper A"), _scored("RAW-B", "Paper B")]

        with self.assertRaisesRegex(LLMNoFallbackError, "unsafe activity"):
            _summarize_chunk(object(), _settings(), scored_paperid)

        chatMock.assert_called_once()

    @patch("freshlit.nodes.synthesis.chat_structured")
    def test_no_fallback_error_from_single_paper_propagates(self, chatMock):
        chatMock.side_effect = LLMNoFallbackError("unsafe activity")

        with self.assertRaisesRegex(LLMNoFallbackError, "unsafe activity"):
            summarize_paper(object(), _settings(), _scored("RAW-A", "Paper A"))

        chatMock.assert_called_once()

    @patch("freshlit.nodes.synthesis.chat_structured")
    def test_no_fallback_error_from_field_pulse_propagates(self, chatMock):
        chatMock.side_effect = LLMNoFallbackError("unsafe activity")
        scored = _scored("RAW-A", "Paper A")
        extraction = PaperExtraction(
            core_question="question",
            framework_and_method="method",
            key_finding="finding",
            code_data_link="None stated",
        )
        with patch("freshlit.nodes.synthesis.chat_structured", return_value=extraction):
            summary = summarize_paper(object(), _settings(), scored)
        self.assertIsNotNone(summary)

        with self.assertRaisesRegex(LLMNoFallbackError, "unsafe activity"):
            generate_field_pulse(object(), _settings(), [summary])

        chatMock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
