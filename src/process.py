import multiprocessing as mp
import re

import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from string import punctuation


HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(
    r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#…])*"
)
HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@(\w+)")
RETWEET_RE = re.compile(r"RT @(\w+):")
COD_RE = re.compile(r"COD _ (\w+) ")
BULLET_RE = re.compile(r"^(\d)+.\s")
THREE_DASH_RE = re.compile(r"---.*---")
MORE_THAN_THREE_POINTS_RE = re.compile(r"\.{4,}")


TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

INVALID_START = [
    "List of recent changes",
    "Sort by",
    "Home |",
    "> Home",
    "useful tips",
    "Licenses:",
    "Search in: ",
    "Terms of Use - ",
    "Home page",
    "Home Page",
    "Copyright",
    "Results/Page",
]

INVALID_MIDDLE = [
    " @ ",
    " / ",
    " | ",
    "[...]",
    "(...)",
]


INVALID_END = [
    " (",
    "…",
    "[…]",
    "(…)",
]


MIN_N_TOKENS = 10
MAX_N_TOKENS = 900

MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


def _n_tokens(text):
    return len(TOKENIZER.encode(text))


def bad_translation(pt: str, en: str):
    if 0.8 * len(pt) < len(en) < 1.2 * len(pt):
        return False
    return True


def valid_n_tokens(text, min_n_tokens=MIN_N_TOKENS, max_n_tokens=MAX_N_TOKENS):
    return min_n_tokens <= _n_tokens(text) <= max_n_tokens


def has_hashtag(text):
    return len(HASHTAG_RE.findall(text)) > 0


def has_mention(text):
    return len(MENTION_RE.findall(text)) > 0


def has_url(text):
    return len(URL_RE.findall(text)) > 0


def starts_with_month(text):
    return text.lower().startswith(tuple(MONTHS))


def has_too_long_word(text):
    return any(word for word in text.split(" ") if len(word) > 20)


def has_invalid_start(text):
    return text.startswith(tuple(INVALID_START))


def has_invalid_middle(text):
    return any(True for word in INVALID_MIDDLE if word in text)


def has_invalid_end(text):
    return text.endswith(tuple(INVALID_END))


def has_html_tags(text):
    return bool(HTML_RE.search(text))


def has_more_than_three_points(text):
    return bool(MORE_THAN_THREE_POINTS_RE.search(text))


def has_valid_brackets(text):
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
        and text.count("{") == text.count("}")
    )


def is_empty(text):
    return len(text) == 0


def has_invalid_character(text):
    return any(char for char in text if not char.isalnum() and char not in punctuation)


def huggingface_dataset_filter(dataset):
    return dataset.filter(
        lambda x: valid_n_tokens(f"{x['pt']} {x['en']}")
        and not bad_translation(x["pt"], x["en"])
        and not starts_with_month(x["en"])
        and not has_too_long_word(x["en"])
        and not has_too_long_word(x["pt"])
        and not has_invalid_start(x["en"])
        and not has_invalid_middle(x["en"])
        and not has_invalid_end(x["en"])
        and not has_invalid_end(x["pt"])
        and not has_more_than_three_points(x["en"])
        and not has_more_than_three_points(x["pt"])
        and not is_empty(x["pt"])
        and has_invalid_character(x["pt"])
        and has_valid_brackets(x["en"]),
        num_proc=mp.cpu_count(),
    )


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_hashtags(text):
    return HASHTAG_RE.sub("", text).strip()


def remove_mentions(text):
    return MENTION_RE.sub("", text).strip()


def remove_retweets(text):
    return RETWEET_RE.sub("", text).strip()


def remove_urls(text):
    return URL_RE.sub("", text).strip()


def remove_cod_literature(text):
    return COD_RE.sub("", text).strip()


def remove_bullets(text):
    return BULLET_RE.sub("", text).strip()


def remove_three_dashes(text):
    return THREE_DASH_RE.sub("", text).strip()


def huggingface_dataset_transform(dataset):
    def _transform(text):
        text = remove_retweets(text)
        text = remove_mentions(text)
        text = remove_hashtags(text)
        text = remove_urls(text)
        text = remove_html_tags(text)
        text = remove_cod_literature(text)
        text = remove_bullets(text)
        text = remove_three_dashes(text)
        return text

    return dataset.map(
        lambda x: {
            "pt": _transform(x["pt"]),
            "en": _transform(x["en"]),
        },
        num_proc=mp.cpu_count(),
    )


def drop_duplicates(dataset):
    """Drop all the rows that have the same start and end n_chars."""
    _, unique_idxs_train = np.unique(dataset["train"]["pt"], return_index=True, axis=0)
    _, unique_idxs_test = np.unique(dataset["test"]["pt"], return_index=True, axis=0)
    dataset["train"] = dataset["train"].select(list(unique_idxs_train))
    dataset["test"] = dataset["test"].select(list(unique_idxs_test))
    return dataset


def drop_duplicates_start_ends(dataset, n_chars: int = 60):
    """Drop all the rows that have the same start and end n_chars."""

    def get_unique_idxs(dataset):
        _, unique_starts_idxs = np.unique(dataset["start"], return_index=True, axis=0)
        _, unique_ends_idxs = np.unique(dataset["end"], return_index=True, axis=0)
        unique_idxs = set.intersection(set(unique_starts_idxs), set(unique_ends_idxs))
        return unique_idxs

    temp = dataset.map(
        lambda x: {"start": x["pt"][:n_chars], "end": x["pt"][-n_chars:]},
        num_proc=mp.cpu_count(),
    )

    unique_idxs_train = get_unique_idxs(temp["train"])
    unique_idxs_test = get_unique_idxs(temp["test"])
    dataset["train"] = dataset["train"].select(list(unique_idxs_train))
    dataset["test"] = dataset["test"].select(list(unique_idxs_test))
    return dataset
