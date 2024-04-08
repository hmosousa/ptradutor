# Público News Translator

Translate news articles [scraped](https://github.com/hmosousa/publico_scraper) from Jornal Público into multiple languages using Google Translate.

## Getting Started

This guide will help you set up a development environment, download the news data, and translate it into the language of your choice.

### Setting Up Your Development Environment

To begin, you'll need to create a virtual environment and install the necessary dependencies. Follow these steps:

```sh
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the project dependencies
pip install -e .
```

### Downloading the Data

To download the news articles from Jornal Público, execute the following command:

```sh
sh scripts/download.sh
```

### Translating the News

Before translating the news, we first prepare the data by parsing it to remove metadata and eliminate duplicates:

```sh
python scripts/parse.py
```

This process generates a file named `pt.jsonl` in the `data` directory, containing all news articles ready for translation. To translate these articles into a desired language, use:

```sh
python scripts/translate.py --lang=<target_language>
```

Please replace `<target_language>` with the code of the language you want to translate the articles into.

#### Language Support

The translation tool supports multiple languages. To check the list of available languages for translation, run the following command in your terminal:

```sh
list_languages
```

## License

The code of this project is released under the MIT License.

## Contact

For questions, suggestions, or collaborations, please contact the project maintainer:

- **Project Maintainer:** Hugo Sousa
- **Email:** hugo.o.sousa@inesctec.pt
