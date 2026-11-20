"""Step 1: loading the data.

The full CFPB export is a few GB and won't fit in memory comfortably on my
laptop, so I read it in chunks and only keep the three columns I need.
"""

import pandas as pd

TEXT_COL = "Consumer complaint narrative"
LABEL_COLS = ["Product", "Issue"]


def load_sample(path, n_samples=30000, seed=42, chunksize=200_000):
    """Returns a random sample of complaints that actually have text."""
    usecols = [TEXT_COL] + LABEL_COLS
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                             dtype=str, low_memory=False):
        # most rows have no narrative at all (people can submit without one)
        chunk = chunk.dropna(subset=[TEXT_COL])
        chunk = chunk[chunk[TEXT_COL].str.strip() != ""]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"Complaints with a narrative: {len(df):,}")

    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    df = df.rename(columns={TEXT_COL: "text", "Product": "product", "Issue": "issue"})
    print(f"Sample size: {len(df):,}")
    return df


def describe(df):
    """Quick look at the sample before cleaning: lengths, masking, products."""
    lengths = df["text"].str.split().str.len()
    print("\nWords per complaint: "
          f"mean {lengths.mean():.0f}, median {lengths.median():.0f}, "
          f"min {lengths.min()}, max {lengths.max()}")
    masked = df["text"].str.contains("XXXX").mean()
    print(f"Complaints containing masked data XXXX: {masked:.1%}")
    print("\nTop products in the sample:")
    print(df["product"].value_counts(normalize=True).head(8).round(3).to_string())
