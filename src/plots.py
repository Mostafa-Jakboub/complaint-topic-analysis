"""All the plots. Everything gets saved to results/figures."""

import matplotlib

matplotlib.use("Agg")  # no windows popping up, just write the files
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# colours, kept in one place so all figures match
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Only the darker blues for the word clouds, light words are hard to read.
CLOUD = LinearSegmentedColormap.from_list("cloud", ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  saved {path}")


def coherence_curve(scores, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, color in [("LDA", BLUE), ("NMF", ORANGE)]:
        ax.plot(scores["k"], scores[col], color=color, lw=2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=col)
        # draw a ring around the best k
        best = scores.loc[scores[col].idxmax()]
        ax.plot(best["k"], best[col], marker="o", ms=15, mfc="none", mec=color, mew=2)
    lo, hi = scores[["LDA", "NMF"]].min().min(), scores[["LDA", "NMF"]].max().max()
    pad = (hi - lo) * 0.2 or 0.02
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Number of topics k")
    ax.set_ylabel("Coherence c_v")
    k_lda = int(scores.loc[scores["LDA"].idxmax(), "k"])
    k_nmf = int(scores.loc[scores["NMF"].idxmax(), "k"])
    ax.set_title(f"Topic coherence by number of topics (circled = best: LDA k={k_lda}, "
                 f"NMF k={k_nmf})", loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xticks(scores["k"])
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def topic_shares(table, title, path, color=BLUE):
    t = table.sort_values("share")
    def short(words, limit=40):
        # cut the label at ~40 chars (whole words only), long bigrams were squashing the bars
        out = ""
        for w in words.split(", ")[:4]:
            nxt = f"{out}, {w}" if out else w
            if len(nxt) > limit:
                break
            out = nxt
        return out
    labels = [f"T{r.topic}: {short(r.top_words)}" for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.2))
    ax.barh(labels, t["share"] * 100, color=color, height=0.7)
    for y, v in enumerate(t["share"] * 100):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, t["share"].max() * 100 * 1.15)
    ax.set_xlabel("Share of complaints, %")
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_title(title, loc="left", fontsize=12)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _save(fig, path)


def topic_vs_product(dominant, products, path, n_products=8):
    """Heatmap of topics against the CFPB product labels.

    Each row adds up to 100%, so you can read it as "of the complaints in
    topic 3, how many were about mortgages, credit cards, ...". Good sanity
    check that the topics mean something.
    """
    df = pd.DataFrame({"topic": dominant + 1, "product": products})
    # small products get lumped into "Other" or the heatmap gets way too wide
    top = df["product"].value_counts().head(n_products).index
    df.loc[~df["product"].isin(top), "product"] = "Other"
    ct = pd.crosstab(df["topic"], df["product"], normalize="index")
    ct = ct[[c for c in top if c in ct.columns] + (["Other"] if "Other" in ct else [])]

    fig, ax = plt.subplots(figsize=(1.0 * ct.shape[1] + 3, 0.45 * ct.shape[0] + 2))
    im = ax.imshow(ct.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels([c[:28] for c in ct.columns], rotation=40, ha="right", color=INK_2)
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels([f"Topic {t}" for t in ct.index], color=INK_2)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            if v >= 0.1:  # skip labels on tiny cells, too cluttered otherwise
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.5 else INK)
    ax.set_title("NMF topics vs CFPB product labels", loc="left", fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Share of topic's complaints")
    _save(fig, path)


def word_clouds(model, feature_names, path, n_words=30):
    """One small word cloud per topic, laid out in a grid of 4 columns."""
    from wordcloud import WordCloud

    k = model.components_.shape[0]
    cols = 4
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4))
    for i, ax in enumerate(np.ravel(axes)):
        ax.axis("off")
        if i >= k:
            continue
        comp = model.components_[i]
        freqs = {feature_names[j]: comp[j] for j in comp.argsort()[::-1][:n_words]}
        wc = WordCloud(width=400, height=280, background_color=SURFACE,
                       colormap=CLOUD, prefer_horizontal=1.0, random_state=1)
        ax.imshow(wc.generate_from_frequencies(freqs), interpolation="bilinear")
        ax.set_title(f"Topic {i + 1}", fontsize=10, color=INK_2)
    _save(fig, path)