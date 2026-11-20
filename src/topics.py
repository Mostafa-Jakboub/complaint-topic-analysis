"""Step 4: the topic models (LDA and NMF) and how k gets picked."""

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import NMF, LatentDirichletAllocation


def fit_lda(bow_matrix, k, seed=42):
    # online learning is a lot quicker than batch on 30k docs
    model = LatentDirichletAllocation(n_components=k, learning_method="online",
                                      max_iter=15, random_state=seed, n_jobs=-1)
    doc_topic = model.fit_transform(bow_matrix)
    return model, doc_topic


def fit_nmf(tfidf_matrix, k, seed=42):
    # nndsvda starts from an SVD instead of random numbers, usually converges faster
    model = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=seed)
    doc_topic = model.fit_transform(tfidf_matrix)
    return model, doc_topic


def top_words(model, feature_names, n=10):
    """Top n words of each topic."""
    return [[feature_names[i] for i in comp.argsort()[::-1][:n]]
            for comp in model.components_]


def coherence(topics, tokenized, dictionary):
    """c_v coherence, i.e. do the top words of a topic actually show up together.

    gensim's dictionary only knows single words, so bigrams like
    "credit report" get split up first (and duplicates removed).
    """
    flat = [[w for term in t for w in term.split()] for t in topics]
    flat = [list(dict.fromkeys(t))[:10] for t in flat]
    cm = CoherenceModel(topics=flat, texts=tokenized, dictionary=dictionary,
                        coherence="c_v", processes=1)
    return cm.get_coherence()


def search_k(bow, bow_names, tfidf_m, tfidf_names, clean_texts, k_values,
             coherence_docs=5000, seed=42):
    """Fits both models for every k and returns a table of coherence scores."""
    # coherence on all 30k texts is painfully slow, 5k is enough for ranking k
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(clean_texts), size=min(coherence_docs, len(clean_texts)),
                     replace=False)
    tokenized = [clean_texts[i].split() for i in sub]
    dictionary = Dictionary(tokenized)

    rows = []
    for k in k_values:
        lda, _ = fit_lda(bow, k, seed)
        nmf, _ = fit_nmf(tfidf_m, k, seed)
        c_lda = coherence(top_words(lda, bow_names), tokenized, dictionary)
        c_nmf = coherence(top_words(nmf, tfidf_names), tokenized, dictionary)
        rows.append({"k": k, "LDA": c_lda, "NMF": c_nmf})
        print(f"  k={k:>2}  coherence LDA {c_lda:.3f}  NMF {c_nmf:.3f}")
    return pd.DataFrame(rows)


def topic_table(model, doc_topic, feature_names, method):
    """One row per topic with its top words and how many complaints it covers.

    Each complaint is counted under its strongest topic only, which is a
    simplification but makes the shares easy to read.
    """
    dominant = doc_topic.argmax(axis=1)
    share = np.bincount(dominant, minlength=doc_topic.shape[1]) / len(dominant)
    words = top_words(model, feature_names)
    table = pd.DataFrame({
        "method": method,
        "topic": range(1, len(words) + 1),
        "share": share.round(4),
        "top_words": [", ".join(w) for w in words],
    })
    return table.sort_values("share", ascending=False).reset_index(drop=True), dominant


def word_overlap(lda_words, nmf_words):
    """Matches every LDA topic to the NMF topic with the most top words in common.

    Uses Jaccard similarity, so 1 means identical word lists and 0 means none shared.
    """
    rows = []
    for i, a in enumerate(lda_words):
        scores = [len(set(a) & set(b)) / len(set(a) | set(b)) for b in nmf_words]
        j = int(np.argmax(scores))
        rows.append({"lda_topic": i + 1, "best_nmf_topic": j + 1,
                     "jaccard": round(scores[j], 2)})
    return pd.DataFrame(rows)
