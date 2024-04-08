import logging

from fire import Fire
from tqdm import tqdm

from src.data import TranslationDataset, load_dataset
from src.translator import Translator


logging.basicConfig(level=logging.INFO)

def main(lang="en"):
    translator = Translator(source="pt", target=lang)

    idx = 0
    source_ds = TranslationDataset("pt")
    datasets = ["pt_vid", "frmt", "dsl_tl"]  # TODO: add "pt_vid"
    for name in datasets:
        logging.info(f"Translating dataset {name}.")
        dataset = load_dataset(name)
        for domain in dataset:
            for split in dataset[domain]:
                for text in tqdm(dataset[domain][split]):
                    if idx not in source_ds:
                        try:
                            translate = translator.translate(text)
                            data = {
                                "idx": idx,
                                "source": name,
                                "domain": domain,
                                "split": split,
                                "pt": text, 
                                "en": translate
                            }
                            source_ds.add(idx, data)
                        except Exception as e:
                            print(f"Error translating {idx}: {e}")
                    else:
                        logging.info(f"Skipping {idx}.")
                    idx += 1
                    if idx % 100 == 0:
                        source_ds.save()
    source_ds.save()


if __name__ == "__main__":
    Fire(main)
