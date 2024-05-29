import multiprocessing as mp
import re

from bs4 import BeautifulSoup
from transformers import AutoTokenizer

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#…])*")
HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@(\w+)")
RETWEET_RE = re.compile(r"RT @(\w+):")

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

INVALID_START = [
    "List of recent changes",
    "Sort by",
    "Home |",
    "> Home",
    "useful tips",
    "Licenses:",
]

INVALID_MIDDLE = [
    " @ ",
    " / ",
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


def is_unfinished(text):
    return "[...]" in text or "(...)" in text


def is_pagination(text):
    return "Results/Page" in text


def has_too_long_word(text):
    return any(word for word in text.split(" ") if len(word) > 30)


def has_invalid_start(text):
    return text.startswith(tuple(INVALID_START))


def has_invalid_middle(text):
    return any(True for word in INVALID_MIDDLE if word in text)


def has_html_tags(text):
    return bool(HTML_RE.search(text))


def huggingface_dataset_filter(dataset):
    return dataset.filter(
        lambda x: valid_n_tokens(f"{x['pt']} {x['en']}")
        and not starts_with_month(x["pt"])
        and not is_unfinished(x["pt"])
        and not is_pagination(x["pt"])
        and not has_too_long_word(x["pt"])
        and not has_invalid_start(x["pt"])
        and not has_invalid_middle(x["pt"]),
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


def huggingface_dataset_transform(dataset):
    def _transform(text):
        text = remove_retweets(text)
        text = remove_mentions(text)
        text = remove_hashtags(text)
        text = remove_urls(text)
        text = remove_html_tags(text)
        return text

    return dataset.map(
        lambda x: {
            "pt": _transform(x["pt"]),
            "en": _transform(x["en"]),
        },
        num_proc=mp.cpu_count(),
    )
