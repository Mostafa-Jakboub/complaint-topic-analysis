"""Finds the main topics in the CFPB complaint texts.

This script runs everything in order: load a sample, clean the text,
vectorize it (TF-IDF and embeddings), fit LDA and NMF for a few values
of k, then write the tables and plots to results/.

Usage:
    python main.py --input data/complaints.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import load_data, plots, preprocess, topics, vectorize

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
CLEAN_CACHE = Path("data/clean_sample.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/complaints.csv",
                   help="path to the CFPB complaints CSV")
    p.add_argument("--n", type=int, default=30000, help="number of complaints to sample")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=15)
    p.add_argument("--k-step", type=int, default=2)
    p.add_argument("--embed-n", type=int, default=5000,
                   help="complaints to embed for the vectorization comparison")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="skip the sentence-embedding step, e.g. without internet")
    p.add_argument("--refresh", action="store_true",
                   help="ignore the cached clean sample and preprocess again")
    return p.parse_args()


def step(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = {}

    # Steps 1 and 2. spaCy takes ages on 30k texts, so I save the cleaned
    # sample once and reuse it unless --refresh is passed.
    if CLEAN_CACHE.exists() and not args.refresh:
        step("1-2. Loading cached clean sample")
        df = pd.read_csv(CLEAN_CACHE, dtype=str)
        print(f"Loaded {len(df):,} preprocessed complaints from {CLEAN_CACHE}")
    else:
        step("1. Loading data")
        df = load_data.load_sample(args.input, args.n, args.seed)
        load_data.describe(df)
        step("2. Preprocessing")
        df = preprocess.preprocess(df)
        df.to_csv(CLEAN_CACHE, index=False)
    summary["n_complaints"] = len(df)
    print("\nExample before / after cleaning:")
    print("  RAW:  ", df["text"].iloc[0][:200].replace("\n", " "), "...")
    print("  CLEAN:", df["clean"].iloc[0][:200], "...")

    # Step 3
    step("3. Vectorization")
    bow_vec, bow = vectorize.bag_of_words(df["clean"])
    tfidf_vec, tfidf_m = vectorize.tfidf(df["clean"])
    bow_names = bow_vec.get_feature_names_out()
    tfidf_names = tfidf_vec.get_feature_names_out()
    print(f"Bag-of-Words matrix: {bow.shape[0]:,} x {bow.shape[1]:,}")
    print(f"TF-IDF matrix:       {tfidf_m.shape[0]:,} x {tfidf_m.shape[1]:,}")

    idf = pd.Series(tfidf_vec.idf_, index=tfidf_names)
    print("Most common terms (lowest IDF):", ", ".join(idf.nsmallest(10).index))

    # Embedding all 30k texts on a CPU is slow, so only a random subset is
    # embedded. It's only used for the comparison, not for the topic models.
    if not args.skip_embeddings:
        rng = np.random.default_rng(args.seed)
        rows = rng.choice(len(df), size=min(args.embed_n, len(df)), replace=False)
        emb = vectorize.embeddings(df["text"].iloc[rows].tolist())
        print(f"Embedding matrix:    {emb.shape[0]:,} x {emb.shape[1]}")
        summary["vectorization"] = vectorize.compare(tfidf_m, emb, df["product"], rows)

    # Step 4: try every k first, then refit once with the best one per model
    step("4. Choosing the number of topics")
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    scores = topics.search_k(bow, bow_names, tfidf_m, tfidf_names,
                             df["clean"].tolist(), k_values, seed=args.seed)
    scores.to_csv(RESULTS / "coherence_scores.csv", index=False)
    plots.coherence_curve(scores, FIGURES / "coherence_vs_k.png")
    k_lda = int(scores.loc[scores["LDA"].idxmax(), "k"])
    k_nmf = int(scores.loc[scores["NMF"].idxmax(), "k"])
    summary["best_k"] = {"LDA": k_lda, "NMF": k_nmf}
    summary["best_coherence"] = {"LDA": round(scores["LDA"].max(), 3),
                                 "NMF": round(scores["NMF"].max(), 3)}

    step(f"5. Final models: LDA k={k_lda}, NMF k={k_nmf}")
    lda, lda_dt = topics.fit_lda(bow, k_lda, args.seed)
    nmf, nmf_dt = topics.fit_nmf(tfidf_m, k_nmf, args.seed)
    lda_table, _ = topics.topic_table(lda, lda_dt, bow_names, "LDA")
    nmf_table, nmf_dom = topics.topic_table(nmf, nmf_dt, tfidf_names, "NMF")
    pd.concat([lda_table, nmf_table]).to_csv(RESULTS / "topics.csv", index=False)

    for name, table in [("LDA", lda_table), ("NMF", nmf_table)]:
        print(f"\n{name} topics, largest first:")
        for r in table.itertuples():
            print(f"  T{r.topic:<3}{r.share:>6.1%}  {r.top_words}")

    # Do the two models find roughly the same topics? Check by top-word overlap.
    overlap = topics.word_overlap(topics.top_words(lda, bow_names),
                                  topics.top_words(nmf, tfidf_names))
    overlap.to_csv(RESULTS / "lda_nmf_overlap.csv", index=False)
    summary["mean_lda_nmf_jaccard"] = round(overlap["jaccard"].mean(), 2)
    print(f"\nMean top-word overlap LDA vs best NMF match: "
          f"{summary['mean_lda_nmf_jaccard']:.2f}")

    plots.topic_shares(lda_table, f"LDA topics, k={k_lda}", FIGURES / "topics_lda.png")
    plots.topic_shares(nmf_table, f"NMF topics, k={k_nmf}", FIGURES / "topics_nmf.png")
    plots.topic_vs_product(nmf_dom, df["product"].values, FIGURES / "nmf_topics_vs_product.png")
    plots.word_clouds(nmf, tfidf_names, FIGURES / "nmf_wordclouds.png")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    step("Done. Results are in the results/ folder.")


if __name__ == "__main__":
    main()
