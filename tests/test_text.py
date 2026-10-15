"""Stage 3's text features: the vocabulary counts creators, empty text is zeros, and the embedding cache holds."""

import numpy as np
import pytest

from sharerate import text


def test_clean_drops_mentions_and_links():
    assert text.clean("Big NEWS @some.account see https://x.co/abc and www.y.com").split() == ["big", "news", "see",
                                                                                               "and"]


def test_vocabulary_counts_creators_not_videos():
    # "signature" is used 100 times by one creator; "shared" once by each of 20 creators; "rare" by 19.
    texts = ["signature"] * 100 + [f"shared word{c}" for c in range(20)] + ["rare"] * 19
    creators = [0] * 100 + list(range(1, 21)) + list(range(1, 20))
    vocabulary = text.creator_vocabulary(texts, creators, min_creators=20)
    assert vocabulary.to_dict() == {"shared": 20}


def test_vocabulary_holds_word_pairs():
    texts = ["wait for it"] * 20
    vocabulary = text.creator_vocabulary(texts, range(20), min_creators=20)
    assert set(vocabulary.index) == {"wait", "for", "it", "wait for", "for it"}


def test_no_term_reaches_the_threshold():
    vectorizer, vocabulary = text.fit_tfidf(["only me"] * 5, [0] * 5, min_creators=20)
    assert vectorizer is None and vocabulary.empty


def test_empty_or_unknown_text_is_a_row_of_zeros():
    texts = [f"good joke {c}" for c in range(20)]
    vectorizer, _ = text.fit_tfidf(texts, range(20), min_creators=20)
    rows = vectorizer.transform(["", "nothing in the vocabulary", "good joke"]).toarray()
    assert not rows[0].any() and not rows[1].any()
    assert np.isclose(np.linalg.norm(rows[2]), 1)  # every non-empty row has length 1


def fake_encoder(calls):
    def encode(batch):
        calls.append(list(batch))
        # A deterministic unit vector per text, from its length and first letter.
        vectors = np.array([[len(t), ord(t[0]), 1.0] for t in batch])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return encode


def test_embed_caches_each_distinct_text_once(tmp_path):
    calls = []
    first = text.embed(["abc", "", "abc", "de"], "fake/model", encoder=fake_encoder(calls), cache_dir=tmp_path,
                       chunk=1)
    assert sorted(t for batch in calls for t in batch) == ["abc", "de"]  # "abc" once, "" never
    assert not first[1].any()  # empty text: zeros
    assert np.allclose(first[0], first[2])
    assert np.allclose(np.linalg.norm(first[[0, 3]], axis=1), 1)
    assert len(list((tmp_path / "fake--model").glob("part-*.npz"))) == 2  # one part per chunk

    calls.clear()
    again = text.embed(["de", "abc", "fgh"], "fake/model", encoder=fake_encoder(calls), cache_dir=tmp_path)
    assert calls == [["fgh"]]  # only the new text is embedded
    assert np.allclose(again[0], first[3]) and np.allclose(again[1], first[0])


def test_an_interrupted_part_is_ignored(tmp_path):
    folder = tmp_path / "fake--model"
    folder.mkdir()
    np.savez(folder / "part-00000.tmp.npz", keys=np.array([b"0" * 32]), vectors=np.ones((1, 3)))
    calls = []
    text.embed(["abc"], "fake/model", encoder=fake_encoder(calls), cache_dir=tmp_path)
    assert calls == [["abc"]]


@pytest.mark.parametrize("field", list(text.FIELDS))
def test_every_field_is_a_loader_column(field):
    from sharerate import data
    assert field in data.TEXT_COLUMNS
