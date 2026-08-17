import json
from pathlib import Path
import subprocess
import sys
import unittest

from smalllm import AdvancedAssistant, AssistantSettings
from smalllm.assistant import generation_is_grounded
from smalllm.backend import GenerationSettings, ScratchTokenizer, ScratchTransformerBackend
from smalllm.config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CORPUS_PATH,
    DEFAULT_RETRIEVAL_INDEX,
    DEFAULT_RETRIEVAL_CORPUS_PATH,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_TRAINING_REPORT,
    PROJECT_ROOT,
)
from smalllm.retrieval import TrainedRetriever, load_corpus
from smalllm.tools import run_tool


class ToolTests(unittest.TestCase):
    def test_arithmetic(self) -> None:
        result = run_tool("What is 19 * (4 + 2)?")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "calculator")
        self.assertEqual(result.text, "The result is 114.")

    def test_unit_conversion(self) -> None:
        result = run_tool("Convert 32 fahrenheit to celsius")
        self.assertIsNotNone(result)
        self.assertIn("0 c", result.text)

    def test_text_statistics(self) -> None:
        result = run_tool("Count words in: a small useful model")
        self.assertEqual(result.text, "Words: 4; characters: 20; lines: 1.")

    def test_json_validation(self) -> None:
        valid = run_tool('Validate JSON: {"ready": true}')
        invalid = run_tool('Validate JSON: {"ready": nope}')
        self.assertIn('"ready": true', valid.text)
        self.assertIn("Invalid JSON", invalid.text)

    def test_arbitrary_python_is_never_executed(self) -> None:
        self.assertIsNone(run_tool("calculate __import__('os').getcwd()"))

    def test_reschedule_email_draft(self) -> None:
        result = run_tool("Write a polite email asking to reschedule a meeting")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "email_draft")
        self.assertIn("Request to Reschedule", result.text)

    def test_guided_programming_and_http_responses(self) -> None:
        python_result = run_tool(
            "Write a Python function that removes duplicate strings while preserving order"
        )
        http_result = run_tool("What should I check when debugging an HTTP 500 error?")
        self.assertEqual(python_result.name, "python_help")
        self.assertIn("dict.fromkeys", python_result.text)
        self.assertEqual(http_result.name, "http_troubleshooting")
        self.assertIn("server/application logs", http_result.text)


class CorpusAndRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_corpus(DEFAULT_CORPUS_PATH)
        cls.retrieval_records = load_corpus(DEFAULT_RETRIEVAL_CORPUS_PATH)
        cls.retriever = TrainedRetriever(
            DEFAULT_RETRIEVAL_CORPUS_PATH, DEFAULT_RETRIEVAL_INDEX
        )

    def test_expanded_corpus_size_and_sources(self) -> None:
        self.assertEqual(len(self.records), 11_262)
        sources = {record["source"] for record in self.records}
        self.assertEqual(
            sources,
            {
                "smalllm_curated",
                "OpenAssistant/oasst1",
                "OpenAssistant/oasst2",
            },
        )
        self.assertGreaterEqual(len(self.retrieval_records), 25_000)
        self.assertIn(
            "databricks/databricks-dolly-15k",
            {record["source"] for record in self.retrieval_records},
        )

    def test_common_queries_find_relevant_answers(self) -> None:
        cases = {
            "difference between weather and climate": "weather",
            "how to recognize a phishing message": "phishing",
            "write a professional email": "email",
            "what is photosynthesis": "photosynthesis",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                result = self.retriever.search(query, 1)[0]
                combined = f"{result.prompt} {result.response}".lower()
                self.assertIn(expected, combined)
                self.assertGreater(result.score, 0.30)

    def test_hybrid_assistant_uses_tools_and_retrieval(self) -> None:
        assistant = AdvancedAssistant(AssistantSettings(backend="retrieval"))
        calculation = assistant.respond("Calculate 7 times 8")
        answer = assistant.respond("How do I recognize phishing?")
        self.assertEqual(calculation.backend, "tool:calculator")
        self.assertEqual(calculation.text, "The result is 56.")
        self.assertEqual(answer.backend, "retrieval")
        self.assertTrue(
            {"credentials", "sender", "link"}.intersection(answer.text.lower().split())
        )

    def test_chat_memory_keeps_recent_turns(self) -> None:
        assistant = AdvancedAssistant(AssistantSettings(backend="retrieval"))
        assistant.respond("Hello")
        assistant.respond("What is JSON?")
        self.assertEqual(len(assistant.history), 4)
        self.assertEqual(assistant.history[0]["role"], "user")
        self.assertEqual(assistant.history[-1]["role"], "assistant")

    def test_generation_grounding_gate(self) -> None:
        self.assertTrue(
            generation_is_grounded(
                "Explain photosynthesis",
                "Photosynthesis lets plants use light to make stored chemical energy.",
            )
        )
        self.assertFalse(
            generation_is_grounded(
                "Explain photosynthesis", "The game is a common type of music."
            )
        )


class FromScratchModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = ScratchTokenizer(DEFAULT_TOKENIZER_PATH)
        cls.backend = ScratchTransformerBackend(
            settings=GenerationSettings(max_new_tokens=12, temperature=0), seed=11
        )

    def test_tokenizer_was_trained_and_has_special_tokens(self) -> None:
        self.assertEqual(self.tokenizer.vocab_size, 8_000)
        self.assertIsInstance(self.tokenizer.token_to_id("<assistant>"), int)
        encoded = self.tokenizer.encode("SmallLM answers questions.")
        self.assertGreater(len(encoded), 2)
        self.assertIn("SmallLM", self.tokenizer.decode(encoded))

    def test_checkpoint_has_expected_architecture(self) -> None:
        self.assertEqual(self.backend.parameter_count(), 6_836_224)
        self.assertEqual(self.backend.config.n_layers, 6)
        self.assertEqual(self.backend.config.n_heads, 8)
        self.assertEqual(self.backend.config.max_seq_len, 192)

    def test_generation_is_reproducible_and_contains_no_control_tokens(self) -> None:
        prompt = "<bos><system>\nYou are helpful.\n<user>\nHello\n<assistant>\n"
        first = self.backend.generate(prompt)
        second_backend = ScratchTransformerBackend(
            settings=GenerationSettings(max_new_tokens=12, temperature=0), seed=11
        )
        second = second_backend.generate(prompt)
        self.assertEqual(first, second)
        self.assertTrue(first.strip())
        self.assertNotIn("<assistant>", first)

    def test_training_report_proves_random_initialization(self) -> None:
        report = json.loads(DEFAULT_TRAINING_REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["trained_from_scratch"])
        self.assertFalse(report["pretrained_weights_used"])
        self.assertEqual(report["corpus_records"], 11_262)
        self.assertGreaterEqual(report["steps"], 7_000)


class ProjectTests(unittest.TestCase):
    def test_cli_retrieval_smoke(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "ask",
                "--backend",
                "retrieval",
                "--prompt",
                "Hello",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertTrue(result.stdout.strip())

    def test_project_stays_under_two_gibibytes(self) -> None:
        total = sum(
            path.stat().st_size
            for path in PROJECT_ROOT.rglob("*")
            if path.is_file()
        )
        self.assertLess(total, 2 * 1024**3)


if __name__ == "__main__":
    unittest.main()
