import json
from pathlib import Path
import tempfile
import unittest

from main import CharacterNGramLanguageModel, QueryAssistantModel


class CharacterNGramLanguageModelTests(unittest.TestCase):
    def test_seeded_generation_is_reproducible(self) -> None:
        model = CharacterNGramLanguageModel(order=3)
        model.train("red bird blue bird green bird")

        first = model.generate("bird", length=40, temperature=0.8, seed=12)
        second = model.generate("bird", length=40, temperature=0.8, seed=12)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len("bird") + 40)

    def test_unknown_context_uses_backoff(self) -> None:
        model = CharacterNGramLanguageModel(order=4)
        model.train("abcabcabc")

        result = model.generate("totally unseen", length=20, seed=4)

        self.assertEqual(len(result), len("totally unseen") + 20)
        self.assertTrue(set(result[-20:]).issubset(set("abc")))

    def test_save_and_load_preserve_output(self) -> None:
        model = CharacterNGramLanguageModel(order=2)
        model.train("one fish two fish")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = CharacterNGramLanguageModel.load(path)

        self.assertEqual(model.to_dict(), loaded.to_dict())
        self.assertEqual(
            model.generate(length=25, seed=3),
            loaded.generate(length=25, seed=3),
        )

    def test_invalid_model_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"version": 99}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unsupported character model"):
                CharacterNGramLanguageModel.load(path)


class QueryAssistantModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = QueryAssistantModel.load("model.json")

    def assert_topic(self, query: str, expected_topic: str) -> None:
        _, topic, score = self.model.answer(query)
        self.assertEqual(topic, expected_topic, msg=f"query={query!r}, score={score}")

    def test_conversation_query(self) -> None:
        answer, topic, score = self.model.answer("Hello")

        self.assertEqual(topic, "greeting")
        self.assertGreaterEqual(score, 0.9)
        self.assertIn("help", answer.lower())

    def test_varied_queries_match_intended_topics(self) -> None:
        cases = {
            "How does a plant turn light into energy?": "photosynthesis",
            "Can you help me debug a script?": "debugging",
            "Explain neural attention models": "transformer",
            "How can I organize a complicated project?": "task_planning",
            "Why should passwords be different?": "passwords",
            "What molecule carries inherited traits?": "dna",
            "Write a concise business email": "professional_email",
        }

        for query, topic in cases.items():
            with self.subTest(query=query):
                self.assert_topic(query, topic)

    def test_arithmetic_uses_calculator(self) -> None:
        answer, topic, score = self.model.answer("What is 19 * (4 + 2)?")

        self.assertEqual(answer, "The result is 114.")
        self.assertEqual(topic, "calculator")
        self.assertEqual(score, 1.0)

    def test_unknown_query_does_not_invent_an_answer(self) -> None:
        answer, topic, score = self.model.answer(
            "How do I calibrate a quantum magnetometer?"
        )

        self.assertIsNone(topic)
        self.assertLess(score, 0.24)
        self.assertIn("do not have enough", answer)

    def test_response_selection_is_reproducible(self) -> None:
        first = self.model.answer("Tell me a joke", seed=8)
        second = self.model.answer("Tell me a joke", seed=8)

        self.assertEqual(first, second)

    def test_hybrid_save_and_load_preserve_answers(self) -> None:
        instructions = [
            {
                "topic": "greeting",
                "prompts": ["hello", "good morning"],
                "responses": ["Hello!", "Good morning!"],
            },
            {
                "topic": "python",
                "prompts": ["what is python", "explain python programming"],
                "responses": ["Python is a programming language."],
            },
        ]
        model = QueryAssistantModel(order=2)
        model.train("A tiny training corpus.", instructions)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assistant.json"
            model.save(path)
            loaded = QueryAssistantModel.load(path)

        self.assertEqual(model.to_dict(), loaded.to_dict())
        self.assertEqual(model.answer("hello", seed=2), loaded.answer("hello", seed=2))


if __name__ == "__main__":
    unittest.main()
