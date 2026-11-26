# Topic Extraction from Consumer Complaints

This is my portfolio project for **Project: Data Analysis (DLBDSEDA02)** at IU International
University, Task 1.

The task: a city wants to find out what its citizens complain about most. I don't have access
to a real municipal complaint system, so I used the **CFPB Consumer Complaint Database**
instead. It's a big public dataset of complaints people sent to a US authority, and most of
them come with a free-text description, which is what I needed.

In short, the code takes a sample of those texts, cleans them up, turns them into vectors
(two different ways, as the task asks), and then runs two topic models (LDA and NMF) to see
which topics come out.

## How it works

1. **Load.** The CSV is huge, so it's read in chunks. I only keep complaints that actually
   have text and take a random sample of 30,000.
2. **Clean.** Lowercase everything, remove the `XXXX` placeholders the CFPB uses to hide
   personal data, plus URLs, numbers and punctuation. Then spaCy lemmatizes the words and
   throws out stopwords. I added my own list of extra stopwords like "bank" or "account"
   because they're in almost every complaint. Texts with fewer than 10 words left and
   duplicates are dropped.
3. **Vectorize.** Two approaches:
   - Bag-of-Words counts and TF-IDF (single words and word pairs, top 5,000 terms)
   - Sentence embeddings from `all-MiniLM-L6-v2` (384 numbers per complaint)

   To compare them I check, for each complaint, whether its 5 closest neighbours are about
   the same product. If they are, the vectors are probably capturing something real.
4. **Topics.** LDA runs on the word counts, NMF on TF-IDF. Both are tried with 5, 7, 9, 11,
   13 and 15 topics, and I keep whichever number gets the best c_v coherence score.
5. **Results.** Tables and plots go into `results/`.

Main libraries: pandas, spaCy, scikit-learn, sentence-transformers, gensim (for coherence),
matplotlib and wordcloud.

## Folder layout

```
complaint-topic-analysis/
├── main.py              run this, it does everything
├── requirements.txt
├── src/
│   ├── load_data.py     loading + sampling
│   ├── preprocess.py    cleaning + lemmatization
│   ├── vectorize.py     BoW, TF-IDF, embeddings and the comparison
│   ├── topics.py        LDA, NMF, coherence
│   └── plots.py         all the figures
├── data/                the CFPB CSV goes here (not on GitHub, too big)
└── results/             output tables and figures
```

## Running it

Clone the repo and make a virtual environment:

```bash
git clone https://github.com/Mostefa-Jakboub/complaint-topic-analysis.git
cd complaint-topic-analysis
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Mac/Linux),
then install everything:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Next you need the data. Download the complaints CSV from consumerfinance.gov (there are also
copies on Kaggle) and save it as `data/complaints.csv`. It needs the columns
`Consumer complaint narrative`, `Product` and `Issue`.

Then:

```bash
python main.py --input data/complaints.csv
```

Be warned, the full run took me around 30 to 45 minutes on my laptop. Most of that is spaCy
and fitting the models over and over for each k. After the first run the cleaned texts are
saved to `data/clean_sample.csv`, so the next runs skip loading and cleaning. Pass
`--refresh` if you want to redo it.

If you just want to see if everything works, try a smaller run first:

```bash
python main.py --n 5000 --k-max 9
```

Other options:

| Option | Default | What it does |
|---|---|---|
| `--n` | 30000 | sample size |
| `--seed` | 42 | random seed, so you get the same results as me |
| `--k-min`, `--k-max`, `--k-step` | 5, 15, 2 | which numbers of topics to try |
| `--embed-n` | 5000 | how many complaints get embedded for the comparison |
| `--skip-embeddings` | off | skips the embedding part (handy if you're offline, since the model gets downloaded) |

## What you get

Everything ends up in `results/`:

- `topics.csv` has every LDA and NMF topic with its top 10 words and what share of complaints
  fall into it
- `coherence_scores.csv` has the coherence for each k, for both models
- `lda_nmf_overlap.csv` matches each LDA topic to its closest NMF topic
- `summary.json` has the main numbers from the run
- `figures/` has the plots: coherence vs k, topic sizes, a heatmap of NMF topics against the
  CFPB product labels, and word clouds

## Data

Consumer Financial Protection Bureau, Consumer Complaint Database,

https://files.consumerfinance.gov/f/documents/CCDB_Export_9_January_2025_through_February_2025.zip
