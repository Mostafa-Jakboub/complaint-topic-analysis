"""Step 2: cleaning and lemmatizing the complaint texts."""

import re

import spacy

# Extra stopwords on top of spaCy's list. These show up in nearly every
# complaint ("bank", "account", "call"...) so they end up in every topic
# and just make the topics harder to tell apart.
DOMAIN_STOPWORDS = {
    "bank", "account", "company", "complaint", "consumer", "customer",
    "would", "could", "also", "get", "go", "say", "tell", "call", "ask",
    "time", "day", "month", "year", "know", "receive", "send", "make",
    "take", "want", "need", "letter", "information", "contact", "state",
    "please", "help", "back", "since", "still", "even", "never", "well",
    "problem", "issue",
}

NEG_RE = re.compile(r"n['’]t\b")                 # don't -> do not
APOS_RE = re.compile(r"['’](s|re|ve|ll|d|m)\b")   # it's, they're, I've: just drop the ending
MASK_RE = re.compile(r"x{2,}", re.IGNORECASE)        # CFPB hides names/dates as XXXX
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_RE = re.compile(r"[^a-z\s]")               # digits, punctuation, $ etc.
SPACE_RE = re.compile(r"\s+")


def basic_clean(text):
    """Regex cleanup before spaCy sees the text."""
    text = text.lower()
    # order matters a bit here: fix "n't" before the apostrophe rule eats it
    text = NEG_RE.sub(" not", text)
    text = APOS_RE.sub("", text)
    text = MASK_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NON_LETTER_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def load_spacy():
    # Only need lemmas, so parser and NER are switched off. Makes it a lot faster.
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.max_length = 2_000_000  # a few complaints are really long
    return nlp


def lemmatize(texts, nlp, batch_size=500):
    """Returns each text as a string of lemmas, stopwords removed."""
    stop = nlp.Defaults.stop_words | DOMAIN_STOPWORDS
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        # check the lemma ("called" -> "call") and the word as written, since
        # some stopwords get lemmatized into something that isn't on the list
        tokens = [t.lemma_ for t in doc
                  if t.is_alpha and len(t) > 2 and t.lemma_ not in stop
                  and t.text not in stop]
        out.append(" ".join(tokens))
    return out


def preprocess(df, min_tokens=10):
    """Adds a 'clean' column and throws out very short texts and duplicates."""
    df = df.copy()
    nlp = load_spacy()
    df["clean"] = lemmatize((basic_clean(t) for t in df["text"]), nlp)

    # Under 10 tokens there's not much left to model. Duplicates are mostly
    # template letters that people send in again and again.
    before = len(df)
    df = df[df["clean"].str.split().str.len() >= min_tokens]
    n_short = before - len(df)
    n_before_dups = len(df)
    df = df.drop_duplicates(subset="clean").reset_index(drop=True)
    n_dups = n_before_dups - len(df)
    print(f"Removed {n_short:,} texts with fewer than {min_tokens} tokens")
    print(f"Removed {n_dups:,} duplicate texts, e.g. the same letter submitted repeatedly")
    print(f"Kept {len(df):,} of {before:,} complaints")
    return df