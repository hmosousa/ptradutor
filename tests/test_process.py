from src.process import (
    has_hashtag,
    has_html_tags,
    has_invalid_start,
    has_mention,
    has_too_long_word,
    has_url,
    is_pagination,
    is_unfinished,
    remove_hashtags,
    remove_html_tags,
    remove_mentions,
    remove_retweets,
    remove_urls,
    starts_with_month,
    valid_n_tokens,
)

def test_valid_n_tokens():
    assert not valid_n_tokens("Hi")


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


def test_is_unfinished():
    assert is_unfinished("This text (...) incomplete")
    assert is_unfinished("This text is [...] incomplete")
    assert not is_unfinished("This text is not incomplete")


def test_is_pagination():
    assert is_pagination("Results/Page")


def test_has_too_long_word():
    assert has_too_long_word("thissentencehasaverylongwordthatshouldbedropped")


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


def test_remove_mentions():
    assert remove_mentions("i am tagging @you") == "i am tagging"
    assert remove_mentions("i am not tagging you") == "i am not tagging you"
    assert remove_mentions("@ArrobaTuga Look, Cristina is") == "Look, Cristina is"
    assert remove_mentions("@_zucacritica @primaguiar I realized...") == "I realized..."


def test_remove_urls():
    assert remove_urls("this is an url https://www.google.com") == "this is an url"
    assert remove_urls("this is an url www.google.com") == "this is an url"
    assert remove_urls("this is an url google.com") == "this is an url"
    assert remove_urls("FC Porto frente ao Manchester City... https://t.co/y5VnxL9…") == "FC Porto frente ao Manchester City..."


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
