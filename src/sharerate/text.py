"""Stage 3's text features: TF-IDF over the words many creators use, and sentence embeddings.

Two rules shape both, fixed before any model was scored:

- **Fitted on the training rows only.** The vocabulary and the IDF weights come from the training cell, so nothing
  about validation's words leaks into the features.
- **A term enters only if at least `MIN_CREATORS` training creators use it.** A word only one creator uses is that
  creator's signature (their name, a catchphrase, a sponsor), and lift is already measured against the creator, so
  it can teach the model little about content and much about who made the video. The same threshold is rule 9's:
  whatever the model learns can be shown, because every term in it is used by at least 20 creators.

Embeddings are cached on disk under `data/embeddings/` (git-ignored), keyed by the md5 of each text, so a field is
embedded once whichever notebook asks for it, and a text shared by two fields or two videos only once.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sharerate.data import ROOT

# Every text field, in the order the notebook shows them: {loader column: label}
FIELDS = {
    "caption": "Caption",
    "transcript_en": "Transcript",
    "hook_en": "Hook",
    "body_en": "Body",
    "payoff_en": "Payoff",
    "onscreen_text_en": "On-screen text",
    "scene_description_en": "Scene description",
}
# What the creator wrote or said, as against what a vision-language model saw in the picture (trap 5)
WORDS = [field for field in FIELDS if field != "scene_description_en"]
MIN_CREATORS = 20  # rule 9: a term must be used by at least this many training creators
NGRAMS = (1, 2)  # words and word pairs

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 33M parameters, 384 dimensions, runs on the laptop's CPU
EMBEDDING_CACHE = ROOT / "data" / "embeddings"
MAX_TOKENS = 256  # longer texts are cut here: 99% of transcripts are under 222 words

_MENTION = re.compile(r"@[\w.]+")
_LINK = re.compile(r"https?://\S+|www\.\S+")


def clean(text: str) -> str:
    """Lowercase, with @mentions and links removed: both name an account, which is never a feature here."""
    return _LINK.sub(" ", _MENTION.sub(" ", text)).lower()


def creator_vocabulary(texts, creators, min_creators=MIN_CREATORS, ngram_range=NGRAMS) -> pd.Series:
    """The words and word pairs that at least `min_creators` different creators use in `texts`.

    Returns {term: how many creators use it}, sorted by term. Pass the training rows only. A term used 500
    times by one creator counts once; a term used once by each of 20 creators is in.
    """
    texts = list(texts)
    # A term 20 creators use appears in at least 20 videos, so min_df=min_creators drops nothing that could pass,
    # and keeps the matrix small. binary: whether a video uses the term, not how often.
    counter = CountVectorizer(preprocessor=clean, ngram_range=ngram_range, min_df=min_creators, binary=True,
                              dtype=np.int32)
    try:
        used_in_video = counter.fit_transform(texts)  # videos x terms, 1 where the video uses the term
    except ValueError:  # no term reaches min_df (sklearn raises rather than return an empty vocabulary)
        return pd.Series(dtype=int, name="creators")
    codes, _ = pd.factorize(np.asarray(creators))
    # creators x videos: row c has a 1 at each of creator c's videos
    owns = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))))
    used_by_creator = (owns @ used_in_video) > 0  # creators x terms: did the creator use the term at all
    n_creators = np.asarray(used_by_creator.sum(axis=0)).ravel()
    terms = counter.get_feature_names_out()
    keep = n_creators >= min_creators
    return pd.Series(n_creators[keep], index=terms[keep], name="creators").sort_index()


def fit_tfidf(texts, creators, min_creators=MIN_CREATORS, ngram_range=NGRAMS):
    """A TF-IDF vectorizer over `creator_vocabulary(texts, creators)`, its IDF fitted on the same texts.

    Pass the training rows only, then `.transform` every row. Settings fixed in advance: words and word pairs,
    term frequency as 1 + log(count) (`sublinear_tf`, so a word said ten times doesn't count ten times as much),
    each row scaled to length 1. An empty text, or one with no term in the vocabulary, becomes a row of zeros.
    Returns (vectorizer, vocabulary); `vectorizer` is None when no term reaches `min_creators`.
    """
    texts = list(texts)
    vocabulary = creator_vocabulary(texts, creators, min_creators, ngram_range)
    if vocabulary.empty:
        return None, vocabulary
    vectorizer = TfidfVectorizer(preprocessor=clean, ngram_range=ngram_range, vocabulary=list(vocabulary.index),
                                 sublinear_tf=True, dtype=np.float32)
    return vectorizer.fit(texts), vocabulary


def embed(texts, model_name=EMBEDDING_MODEL, *, encoder=None, cache_dir=EMBEDDING_CACHE, chunk=20_000,
          progress=None) -> np.ndarray:
    """One unit-length sentence embedding per text, and a row of zeros for an empty text.

    Each distinct text is embedded once and cached in `cache_dir/<model>/part-<random id>.npz` (its md5 and its
    vector), a chunk at a time, so a long run that stops keeps what it finished, and two runs at once never
    overwrite each other's parts. `encoder(list_of_texts)` returns
    the vectors; by default it is the sentence-transformers model on the CPU, read from the local Hugging Face
    cache only. `progress(done, total)` is called after each chunk.
    """
    texts = [str(t) for t in texts]
    folder = Path(cache_dir) / model_name.replace("/", "--")
    vectors = _read_cache(folder)  # {md5: vector}
    keys = [_md5(t) if t else None for t in texts]
    missing = {}  # md5: text, for the distinct non-empty texts not cached yet
    for key, t in zip(keys, texts):
        if key is not None and key not in vectors:
            missing[key] = t
    if missing:
        encoder = encoder or _default_encoder(model_name)
        folder.mkdir(parents=True, exist_ok=True)
        todo = list(missing.items())
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            found = np.asarray(encoder([t for _, t in batch]), dtype=np.float32)
            batch_keys = np.array([k for k, _ in batch], dtype="S32")
            # Written to a temporary name, then renamed, so an interrupted write never leaves a broken part.
            name = f"part-{uuid.uuid4().hex}"
            np.savez(folder / f"{name}.tmp.npz", keys=batch_keys, vectors=found)
            os.replace(folder / f"{name}.tmp.npz", folder / f"{name}.npz")
            vectors.update(zip((k.decode() for k in batch_keys), found))
            if progress:
                progress(min(start + chunk, len(todo)), len(todo))
    width = len(next(iter(vectors.values()))) if vectors else 0
    out = np.zeros((len(texts), width), dtype=np.float32)
    for row, key in enumerate(keys):
        if key is not None:
            out[row] = vectors[key]
    return out


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _read_cache(folder: Path) -> dict:
    vectors = {}
    for path in sorted(folder.glob("part-*.npz")):
        if path.name.endswith(".tmp.npz"):  # left by an interrupted write; its chunk is redone
            continue
        with np.load(path) as part:
            vectors.update(zip((k.decode() for k in part["keys"]), part["vectors"]))
    return vectors


def _default_encoder(model_name):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the model already on this disk; never reach the network
    import torch  # imported here: only embedding needs torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(os.cpu_count())  # every logical core: about 25% faster than torch's default here
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = MAX_TOKENS

    def encode(batch):
        # bge needs no instruction for the texts themselves (only for search queries). normalize: unit length,
        # so a dot product is the cosine similarity and every field's vector has the same scale. Small batches
        # of texts sorted by length waste little on padding: 44 texts a second on this laptop's CPU (stage 3's fill,
        # 550,823 texts in 207 minutes).
        return model.encode(batch, batch_size=16, normalize_embeddings=True, show_progress_bar=False)

    return encode
