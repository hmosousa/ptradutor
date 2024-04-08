from deep_translator import GoogleTranslator

_LANGUAGES = GoogleTranslator().get_supported_languages()


def list_languages():
    print(_LANGUAGES)


class Translator:
    def __init__(self, source, target):
        self._source = source
        self._target = target
        self._engine = GoogleTranslator(source=source, target=target)

    def translate(self, text):
        if len(text) <= 5000:
            return self._translate(text)
        else:
            if "\n" in text:
                return self._split_translate(text, "\n")
            else:
                return self._split_translate(text, ". ")

    def _translate(self, text):
        return self._engine.translate(text)

    def _split_translate(self, text, split):
        parts = []
        for part in text.split(split):
            translated = self.translate(part)
            if translated:
                parts.append(translated)
        return split.join(parts)
