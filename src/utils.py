import re


def contains_html_tags(text):
    """
    Detect if the given text contains HTML tags.

    Args:
    text (str): The input string.

    Returns:
    bool: True if HTML tags are found, False otherwise.
    """
    # Define a regular expression pattern to match HTML tags
    html_tag_pattern = re.compile(r"<[^>]+>")

    # Search for HTML tags in the input string
    match = html_tag_pattern.search(text)

    # Return True if any HTML tags are found, otherwise False
    return bool(match)