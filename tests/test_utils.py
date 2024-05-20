from src.utils import contains_html_tags


def test_contains_html_tags():
    # Test the function with a string that contains HTML tags
    text_with_html_tags = "<p>This is a paragraph.</p>"
    assert contains_html_tags(text_with_html_tags) is True

    # Test the function with a string that contains HTML tags
    text_with_html_tags_middle = """Não há plugins para instalar ou ativar. <a href="%1$s"title="Voltar para o Painel">Voltar para o Painel</a>"""
    assert contains_html_tags(text_with_html_tags_middle) is True

    # Test the function with a string that does not contain HTML tags
    text_without_html_tags = "This is a paragraph."
    assert contains_html_tags(text_without_html_tags) is False
