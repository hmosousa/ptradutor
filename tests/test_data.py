from src.data import load_dsl_tl, load_pt_vid


def test_load_dsl_tl():
    data = load_dsl_tl()
    data = data["default"]
    assert "train" in data
    assert "test" in data
    assert len(data["train"]) == 3_047
    assert len(data["test"]) == 857
    assert isinstance(data["train"][0], str)
    assert isinstance(data["test"][0], str)


def test_load_pt_vid():
    data = load_pt_vid()
    domains = ["journalistic", "legal", "literature", "politics", "social_media", "web"]
    for domain in domains:
        assert "train" in data[domain]
        assert "test" in data[domain]
        assert isinstance(data[domain]["train"][0], str)
        assert isinstance(data[domain]["test"][0], str)
