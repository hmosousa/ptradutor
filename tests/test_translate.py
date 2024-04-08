from pathlib import Path

import pytest

from src.translator import Translator


@pytest.fixture
def long_text():
    cur_dir = Path(__file__).parent
    return (cur_dir / "long_text.txt").read_text()


@pytest.fixture
def long_long_text():
    cur_dir = Path(__file__).parent
    return (cur_dir / "long_text.txt").read_text()


class TestTranslator:
    def test_translate(self):
        translator = Translator(source="pt", target="en")
        text = "Olá, mundo!"
        translated = translator.translate(text)
        assert translated == "Hello World!"

    def test_translate_long(self, long_text):
        translator = Translator(source="pt", target="en")
        translated = translator.translate(long_text)
        assert isinstance(translated, str)

    def test_translate_long_long(self, long_long_text):
        translator = Translator(source="pt", target="en")
        translated = translator.translate(long_long_text)
        assert isinstance(translated, str)
