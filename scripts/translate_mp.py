import concurrent.futures
import logging

from fire import Fire
from tqdm import tqdm

from src.data import TranslationDataset, load_dataset
from src.translator import Translator

logging.basicConfig(level=logging.INFO)


BATCH_SIZE = 1_000


def main(lang="en", name="pt_vid", domain="journalistic", split="train"):
    """
    name = ["pt_vid", "frmt", "dsl_tl"]

    Run the following commands to translate the datasets:

    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "journalistic" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "legal" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "literature" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "politics" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "social_media" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "web" -s "train"
    python scripts/translate_mp.py -l "en" -n "dsl_tl" -d "default" -s "train"
    python scripts/translate_mp.py -l "en" -n "dsl_tl" -d "default" -s "test"
    python scripts/translate_mp.py -l "en" -n "frmt" -d "default" -s "train"
    python scripts/translate_mp.py -l "en" -n "frmt" -d "default" -s "test"
    """
    translator = Translator(source="pt", target=lang)

    def translate(text):
        try:
            translate = translator(text)
            return translate
        except Exception as e:
            logging.error(f"Error translating {text}: {e}")

    logging.info(f"Translating dataset {name}.")
    dataset = load_dataset(name)
    translation_ds = TranslationDataset(f"{lang}_{name}_{domain}_{split}")

    logging.info("Filtering missing translations.")
    texts = dataset[domain][split]
    ids = list(range(len(dataset[domain][split])))
    missing_ids = list(set(ids) - set(map(int, translation_ds.ids)))
    texts = [texts[idx] for idx in missing_ids]

    logging.info("Batching the data.")
    bids = [
        missing_ids[i : i + BATCH_SIZE] for i in range(0, len(missing_ids), BATCH_SIZE)
    ]
    btexts = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    logging.info("Translating texts.")
    for bid, btext in tqdm(zip(bids, btexts), total=len(bids)):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(translate, btext)
            for idx, text, result in zip(bid, btext, results):
                if result:
                    data = {
                        "idx": idx,
                        "source": name,
                        "domain": domain,
                        "split": split,
                        "pt": text,
                        "en": result,
                    }
                    translation_ds.add(idx, data)

        logging.debug("Saving the dataset.")
        translation_ds.save()


if __name__ == "__main__":
    Fire(main)
