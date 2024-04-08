import json
from typing import List, Dict

import datasets

from src.constants import DATA_PATH


class TranslationDataset:
    def __init__(self, name: str) -> None:
        self._name = name
        self._path = DATA_PATH / f"{name}.json"

        if self._path.exists():
            self._data = json.load(self._path.open())
        else:
            self._data = {}

    @property
    def ids(self):
        return list(self._data.keys())

    @property
    def texts(self):
        return list(self._data.values())

    def add(self, id: int, data: dict) -> None:
        self._data[id] = data

    def save(self) -> None:
        json.dump(self._data, self._path.open("w"), ensure_ascii=False, indent=4)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data.items())

    def __contains__(self, id) -> bool:
        return id in self._data


def load_frmt() -> Dict:
    dataset = datasets.load_dataset("LCA-PORVID/frmt")
    test = dataset["test"]
    test = test.filter(lambda x: x["label"] == 0)
    test = test["text"]
    data = {"default": {"test": test}}
    return data


def load_dsl_tl() -> Dict:
    dataset = datasets.load_dataset("LCA-PORVID/dsl_tl")

    test = dataset["test"]
    test = test.filter(lambda x: x["label"] in [0, 1])
    test = test["text"]

    train = dataset["train"]
    train = train.filter(lambda x: x["label"] in [0, 1])
    train = train["text"]
    data = {"default": {"train": train, "test": test}}
    return data


def load_pt_vid() -> Dict:
    data = {}
    domains = ["journalistic", "legal", "literature", "politics", "social_media", "web"]
    for domain in domains:
        dataset = datasets.load_dataset("liaad/PTVId", name=domain)

        train = dataset["train"]
        train = train.filter(lambda x: x["label"] == 0)
        train = train["text"]

        data[domain] = {"train": train, "test": []}
    return data


def load_dataset(name: str) -> List[str]:
    match name:
        case "frmt":
            return load_frmt()
        case "dsl_tl":
            return load_dsl_tl()
        case "pt_vid":
            return load_pt_vid()
        case _:
            raise ValueError(f"Dataset {name} not found")