"""Step 3: vectorizing the texts.

The task asks for two different approaches, so there are two here:
word counts / TF-IDF (sparse, one column per term) and sentence embeddings
from a small pretrained model (dense, 384 numbers per text).
"""

import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def bag_of_words(clean_texts, max_features=5000, min_df=10, max_df=0.5):
    """Plain counts. LDA expects counts, not weights, so this one goes to LDA."""
    vec = CountVectorizer(max_features=max_features, min_df=min_df, max_df=max_df,
                          ngram_range=(1, 2))
    return vec, vec.fit_transform(clean_texts)


def tfidf(clean_texts, max_features=5000, min_df=10, max_df=0.5):
    """TF-IDF for NMF. sublinear_tf stops one rant repeating a word from dominating."""
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df,
                          ngram_range=(1, 2), sublinear_tf=True)
    return vec, vec.fit_transform(clean_texts)


def embeddings(raw_texts, model_name=EMBED_MODEL, batch_size=64):
    """Sentence embeddings with MiniLM.

    Note this uses the raw text, not the cleaned one. The model was trained on
    normal sentences, so stripping stopwords would only hurt it. I just take
    out the XXXX masks.
    """
    # imported here so the rest still runs if sentence-transformers isn't installed
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [re.sub(r"[Xx]{2,}", " ", t) for t in raw_texts]
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)


def label_agreement(matrix, labels, k=5, n_queries=1000, seed=42):
    """How often a complaint's k nearest neighbours have the same product label.

    There's no ground truth for "good vectors", so this is my rough check: if
    the vectors capture meaning, a mortgage complaint's neighbours should
    mostly be mortgage complaints too. Higher is better.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx = rng.choice(matrix.shape[0], size=min(n_queries, matrix.shape[0]), replace=False)
    sims = cosine_similarity(matrix[idx], matrix)
    sims[np.arange(len(idx)), idx] = -1          # don't count a complaint as its own neighbour
    nn = np.argsort(-sims, axis=1)[:, :k]
    return float((labels[nn] == labels[idx][:, None]).mean())


def compare(tfidf_matrix, emb_matrix, labels, emb_rows):
    """Prints TF-IDF vs embeddings side by side and returns the numbers."""
    density = tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
    # only compare on the rows that were embedded, so both get the same complaints
    lab = np.asarray(labels)[emb_rows]
    agree_tfidf = label_agreement(tfidf_matrix[emb_rows], lab)
    agree_emb = label_agreement(emb_matrix, lab)
    rows = {
        "TF-IDF": [f"{tfidf_matrix.shape[1]:,}", f"{density:.2%}", f"{agree_tfidf:.1%}"],
        "Embeddings": [f"{emb_matrix.shape[1]:,}", "100%", f"{agree_emb:.1%}"],
    }
    print(f"\n{'Method':<12}{'Dimensions':>12}{'Non-zero':>11}{'Same product in top-5':>24}")
    for name, (dims, dens, agr) in rows.items():
        print(f"{name:<12}{dims:>12}{dens:>11}{agr:>24}")
    return {"tfidf_agreement": agree_tfidf, "embedding_agreement": agree_emb,
            "tfidf_density": density}
