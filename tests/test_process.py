from src.process import (
    has_hashtag,
    has_html_tags,
    has_invalid_middle,
    has_invalid_start,
    has_mention,
    has_too_long_word,
    has_url,
    remove_bullets,
    remove_cod_literature,
    remove_hashtags,
    remove_html_tags,
    remove_mentions,
    remove_retweets,
    remove_three_dashes,
    remove_urls,
    bad_translation,
    starts_with_month,
    valid_n_tokens,
    has_more_than_three_points,
    has_valid_brackets,
)


def test_valid_n_tokens():
    assert not valid_n_tokens("Hi")
    assert not valid_n_tokens("Crianças. Kids.")


def test_has_hashtag():
    assert has_hashtag("this has an #hashtag.")
    assert not has_hashtag("this does not.")


def test_has_mention():
    assert has_mention("i am tagging @you")
    assert not has_mention("i am not tagging you")


def test_has_url():
    assert has_url("this is an url https://www.google.com")
    assert has_url("this is an url http://www.google.com")
    assert has_url("this is an url www.google.com")
    assert has_url("this is an url google.com")
    assert not has_url("this is not an url google.")
    assert not has_url("this is not an url google. com duas frases")


def test_starts_with_month():
    assert starts_with_month("january is the first month of the year")


def test_has_too_long_word():
    assert has_too_long_word("thissentencehasaverylongwordthatshouldbedropped")
    assert has_too_long_word("thissentencehasaverylong")
    assert not has_too_long_word("thissentencehasavery")


def test_has_invalid_start():
    assert has_invalid_start("List of recent changes")
    assert has_invalid_start("Sort by name")
    assert has_invalid_start("Home |")
    assert has_invalid_start("> Home")
    assert has_invalid_start("useful tips")
    assert has_invalid_start("Licenses:")


def test_has_html_tags():
    assert has_html_tags("<p>This is a paragraph.</p>")
    assert has_html_tags(
        """Não há plugins para instalar ou ativar. <a href="%1$s"title="Voltar para o Painel">Voltar para o Painel</a>"""
    )
    assert not has_html_tags("This is a paragraph.")


def test_remove_hashtags():
    assert remove_hashtags("this has an #hashtag.") == "this has an ."
    assert remove_hashtags("this does not.") == "this does not."
    assert remove_hashtags("#hashtag this does not.") == "this does not."
    assert remove_hashtags("this #hashtag does not.") == "this  does not."


def test_remove_mentions():
    assert remove_mentions("i am tagging @you") == "i am tagging"
    assert remove_mentions("i am not tagging you") == "i am not tagging you"
    assert remove_mentions("@ArrobaTuga Look, Cristina is") == "Look, Cristina is"
    assert remove_mentions("@_zucacritica @primaguiar I realized...") == "I realized..."


def test_remove_urls():
    assert remove_urls("this is an url https://www.google.com") == "this is an url"
    assert remove_urls("this is an url www.google.com") == "this is an url"
    assert remove_urls("this is an url google.com") == "this is an url"
    assert (
        remove_urls("FC Porto frente ao Manchester City... https://t.co/y5VnxL9…")
        == "FC Porto frente ao Manchester City..."
    )


def test_remove_retweets():
    assert remove_retweets("RT @user: this is a retweet") == "this is a retweet"


def test_remove_html_tags():
    assert remove_html_tags("<p>This is a paragraph.</p>") == "This is a paragraph."
    assert (
        remove_html_tags(
            'Não há plugins para instalar ou ativar. <a href="%1$s"title="Voltar para o Painel">Voltar para o Painel</a>'
        )
        == "Não há plugins para instalar ou ativar. Voltar para o Painel"
    )


def test_has_invalid_middle():
    assert has_invalid_middle("List of recent / changes")
    assert has_invalid_middle("List of recent @ changes")


def test_remove_cod_literature():
    assert (
        remove_cod_literature(
            "tanto e dá tanto COD _ L0009P0298X dinheiro que fecham os olhos."
        )
        == "tanto e dá tanto dinheiro que fecham os olhos."
    )
    assert remove_cod_literature("This code shouldnt be removed.") == "This code shouldnt be removed."


def test_remove_bullets():
    assert remove_bullets("1. lugar") == "lugar"
    assert remove_bullets("132. lugar") == "lugar"
    assert (
        remove_bullets("not in the the middle 1. lugar")
        == "not in the the middle 1. lugar"
    )
    assert remove_bullets("1. lugar 2. lugar") == "lugar 2. lugar"
    assert remove_bullets("lugar lugar") == "lugar lugar"


def test_three_dashes():
    assert (
        remove_three_dashes(
            "--- Citação de: naughtykisser em 29 de Novembro de 2010 ---Olá comunidade!"
        )
        == "Olá comunidade!"
    )
    assert (
        remove_three_dashes(
            "--- Citação de: naughtykisser em 29 de Novembro de 2010 ---Olá comunidade!"
        )
        == "Olá comunidade!"
    )
    assert remove_three_dashes("This - should - not - be - removed") == "This - should - not - be - removed"

def test_bad_translation():
    assert bad_translation("This is the original sentence", "short")
    assert bad_translation(
        "This is the original sentence",
        "a very long and unnecessary translation that is just wrong",
    )


def test_has_more_than_three_points():
    assert not has_more_than_three_points("This is a sentence with... three points")
    assert has_more_than_three_points(
        "This is a sentence with........ more than three points"
    )


def test_has_valid_brackets():
    assert not has_valid_brackets("This is a sentence with (an open parenthesis")
    assert not has_valid_brackets("This is a sentence with an open parenthesis)")
    assert has_valid_brackets("This is a sentence with (a closed parenthesis)")
    assert not has_valid_brackets(
        "This is a sentence with (a closed parenthesis) and (an open parenthesis"
    )
    assert has_valid_brackets(
        "This is a sentence with (a closed parenthesis) and (an open parenthesis)"
    )
