"""Semantic similarity between a brief and influencer content.

Default backend: TF-IDF on character n-grams (small, no model download, fits Streamlit's free tier).
Optional backend: sentence-transformers (multilingual MiniLM), enabled with EMBEDDING_BACKEND=sentence-transformers
and `pip install -r requirements-optional.txt`. If it fails to load, we fall back to TF-IDF and say so.

Both backends return cosine similarity in [0, 1] per influencer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, db
from .taxonomy import NICHES, niche_expansion


def influencer_documents(path=None) -> dict[int, str]:
    """bio + organic captions + hashtags + bilingual niche words, one text per influencer."""
    inf = db.read("SELECT id, bio_text FROM influencer", path=path)
    niches = db.read("""SELECT inn.influencer_id, n.name FROM influencer_niche inn
                        JOIN niche n ON n.id = inn.niche_id""", path=path)
    posts = db.read("""SELECT a.influencer_id, c.caption, c.hashtags FROM content_item c
                       JOIN account a ON a.id = c.account_id WHERE c.is_sponsored = 0""", path=path)
    niche_map: dict[int, list[str]] = {}
    for iid, name in zip(niches.influencer_id, niches.name):
        niche_map.setdefault(int(iid), []).append(name)
    post_map: dict[int, list[str]] = {}
    for iid, cap, tags in zip(posts.influencer_id, posts.caption, posts.hashtags.fillna("")):
        post_map.setdefault(int(iid), []).append(f"{cap} {tags}")
    docs = {}
    for iid, bio in zip(inf.id, inf.bio_text):
        iid = int(iid)
        known = [n for n in niche_map.get(iid, []) if n in NICHES]
        docs[iid] = f"{bio} {' '.join(post_map.get(iid, [])[:16])} {niche_expansion(known)}"
    return docs


def brief_document(profile: dict, brief_text: str = "") -> str:
    niches = [n for n in profile.get("relevant_niches", [])]
    parts = [profile.get("summary", ""), " ".join(profile.get("selling_points", [])),
             profile.get("product_name", ""), " ".join(profile.get("tone", [])),
             profile.get("target_audience", {}).get("description", ""), niche_expansion(niches),
             brief_text[:1500]]
    return " ".join(p for p in parts if p)


def _tfidf_similarity(brief_doc: str, docs: dict[int, str]) -> dict[int, float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ids = list(docs)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True, min_df=1)
    mat = vec.fit_transform([docs[i] for i in ids] + [brief_doc])
    sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    return dict(zip(ids, sims.tolist()))


_ST_CACHE: dict = {}


def _st_similarity(brief_doc: str, docs: dict[int, str]) -> dict[int, float]:
    from sentence_transformers import SentenceTransformer
    model = _ST_CACHE.get("model") or SentenceTransformer(config.ST_MODEL)
    _ST_CACHE["model"] = model
    ids = list(docs)
    emb = model.encode([docs[i] for i in ids] + [brief_doc], normalize_embeddings=True, show_progress_bar=False)
    sims = emb[:-1] @ emb[-1]
    return dict(zip(ids, np.clip(sims, 0, 1).tolist()))


def similarity(brief_doc: str, docs: dict[int, str], backend: str | None = None) -> tuple[dict[int, float], str]:
    """Return ({influencer_id: cosine}, backend_name_used)."""
    backend = backend or config.EMBEDDING_BACKEND
    if backend == "sentence-transformers":
        try:
            return _st_similarity(brief_doc, docs), "sentence-transformers"
        except Exception:
            return _tfidf_similarity(brief_doc, docs), "tfidf (sentence-transformers unavailable)"
    return _tfidf_similarity(brief_doc, docs), "tfidf"
