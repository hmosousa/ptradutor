import multiprocessing as mp

import datasets
import plotly.graph_objects as go

from src.process import (
    bad_translation,
    drop_duplicates,
    drop_duplicates_start_ends,
    has_invalid_character,
    has_invalid_end,
    has_invalid_middle,
    has_invalid_start,
    has_more_than_three_points,
    has_too_long_word,
    has_valid_brackets,
    has_valid_quotes,
    huggingface_dataset_transform,
    is_empty,
    starts_with_month,
    valid_n_tokens,
)

N_PROC = mp.cpu_count()
DOMAINS = [
    "default",
    "journalistic",
    "literature",
    "web",
    "politics",
    "legal",
    "social_media",
]


def print_n_docs_by_domain(dataset):
    for domain in DOMAINS:
        ds = dataset.filter(lambda x: x["domain"] == domain, num_proc=N_PROC)
        print(f"{domain:<15} {len(ds)}")


def drop_more_than_900_tokens(dataset):
    return dataset.filter(
        lambda x: valid_n_tokens(f"{x['pt']} {x['en']}"),
        num_proc=mp.cpu_count(),
    )


def drop_patterns(dataset):
    return dataset.filter(
        lambda x: not starts_with_month(x["en"])
        and not has_invalid_start(x["en"])
        and not has_invalid_middle(x["en"])
        and not has_invalid_end(x["en"])
        and not has_invalid_end(x["pt"])
        and not has_more_than_three_points(x["en"])
        and not has_more_than_three_points(x["pt"]),
        num_proc=mp.cpu_count(),
    )


def drop_invalid_chars(dataset):
    return dataset.filter(
        lambda x: not has_invalid_character(x["pt"]),
        num_proc=mp.cpu_count(),
    )


def drop_misc(dataset):
    return dataset.filter(
        lambda x: not bad_translation(x["pt"], x["en"])
        and not has_too_long_word(x["en"])
        and not has_too_long_word(x["pt"])
        and not is_empty(x["pt"])
        and has_valid_brackets(x["pt"])
        and has_valid_quotes(x["pt"]),
        num_proc=mp.cpu_count(),
    )


def compute_stats():
    print("Raw")
    raw = datasets.load_dataset("liaad/PTradutor", "raw")
    print_n_docs_by_domain(raw["train"])
    
    print("\nDuplicates and patterns")
    raw = drop_duplicates(raw)
    raw = drop_duplicates_start_ends(raw)
    raw = huggingface_dataset_transform(raw)
    print_n_docs_by_domain(raw["train"])

    print("\nMax tokens")
    raw = drop_more_than_900_tokens(raw)
    print_n_docs_by_domain(raw["train"])

    print("\nInvalid Chars")
    raw = drop_invalid_chars(raw)
    print_n_docs_by_domain(raw["train"])

    print("\nPatterns")
    raw = drop_patterns(raw)
    print_n_docs_by_domain(raw["train"])

    print("\nMISC")
    raw = drop_misc(raw)
    print_n_docs_by_domain(raw["train"])


def make_plot():
    

    # Define the data
    labels = ["default", "journalistic", "literature", "web", "politics", "legal", "social_media",
            "Raw", "Duplicates and patterns", "Max tokens", "Invalid Chars", "Patterns", "MISC"]

    # Define the source and target nodes
    sources = [0, 1, 2, 3, 4, 5, 6,  # Raw to each category
            0, 1, 2, 3, 4, 5, 6,  # Duplicates and patterns to each category
            0, 1, 2, 3, 4, 5, 6,  # Max tokens to each category
            0, 1, 2, 3, 4, 5, 6,  # Invalid Chars to each category
            0, 1, 2, 3, 4, 5, 6,  # Patterns to each category
            0, 1, 2, 3, 4, 5, 6]  # MISC to each category

    targets = [7, 7, 7, 7, 7, 7, 7,  # Raw to Raw
            8, 8, 8, 8, 8, 8, 8,  # Duplicates and patterns to Duplicates and patterns
            9, 9, 9, 9, 9, 9, 9,  # Max tokens to Max tokens
            10, 10, 10, 10, 10, 10, 10,  # Invalid Chars to Invalid Chars
            11, 11, 11, 11, 11, 11, 11,  # Patterns to Patterns
            12, 12, 12, 12, 12, 12, 12]  # MISC to MISC

    # Define the values for each flow
    values = [1331, 1414884, 22258, 49758, 1771, 461784, 2014752,  # Raw values
            1329, 1386299, 22254, 26690, 1522, 414099, 1961379,  # Duplicates and patterns values
            1329, 1386097, 22254, 17892, 1362, 414068, 1922738,  # Max tokens values
            1268, 1383164, 22218, 15230, 1321, 414067, 1738837,  # Invalid Chars values
            1245, 1330431, 17629, 13657, 1097, 388182, 1686355,  # Patterns values
            1171, 1296965, 17181, 12624, 757, 332851, 1403514]   # MISC values

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        ))])

    # Update layout
    fig.update_layout(title_text="Sankey Diagram for Data Flow", font_size=10)
    fig.show()



if __name__ == "__main__":
    make_plot()
