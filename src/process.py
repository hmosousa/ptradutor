import re

from transformers import AutoTokenizer

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://)?(www\.)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}")
HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"(?<=@)\w+")

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

INVALID_START = [
    "List of recent changes",
    "Sort by",
    "Home |",
    "> Home",
    "useful tips",
    "Licenses:",
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
    # TODO: drop?
    return text.lower().startswith(tuple(MONTHS))


def is_unfinished(text):
    return "[...]" in text or "(...)" in text


def is_pagination(text):
    return "Results/Page" in text


def has_too_long_word(text):
    return any(word for word in text.split(" ") if len(word) > 30)


def has_invalid_start(text):
    return text.startswith(tuple(INVALID_START))


def has_html_tags(text):
    return bool(HTML_RE.search(text))


def huggingface_dataset_filter(dataset):
    return dataset.filter(
        lambda x: valid_n_tokens(f"{x['pt']} {x['en']}")
        and not has_hashtag(x["pt"])
        and not has_mention(x["pt"])
        and not has_url(x["pt"])
        and not starts_with_month(x["pt"])
        and not is_unfinished(x["pt"])
        and not is_pagination(x["pt"])
        and not has_too_long_word(x["pt"])
        and not has_invalid_start(x["pt"])
        and not has_html_tags(x["pt"]),
        num_proc=96
    )
