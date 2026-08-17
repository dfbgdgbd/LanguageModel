import json
from pathlib import Path
import tempfile
import unittest

from main import CharacterNGramLanguageModel


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

            with self.assertRaisesRegex(ValueError, "unsupported model version"):
                CharacterNGramLanguageModel.load(path)


if __name__ == "__main__":
    unittest.main()
