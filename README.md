# BERTweet Semantic Shifting: From Meme to Mainstream

## Overview

This repository contains the code and data for a thesis research project analyzing **semantic shifting** of internet slang words — specifically **"Aura"**, **"Goon"**, and **"Gyat"** — using **BERTweet** embeddings and various NLP techniques. The study tracks how these words have evolved in meaning from their original usage (2015–2019) to their modern internet-era usage (2020–2025) through Twitter/X data.

## Repository Structure

```
├── Data/
│   ├── Aura/
│   │   ├── combined_2015_2019.csv
│   │   └── combined_2020_2025.csv
│   ├── Goon/
│   │   ├── combined_2015_2019.csv
│   │   └── combined_2020_2025.csv
│   └── Gyat/
│       ├── combined_2015_2019.csv
│       └── combined_2020_2025.csv
├── script/
│   └── BERTweet-SemanticShifting.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology

1. **Data Collection** — Tweets scraped from Twitter/X containing target slang words across two time periods (2015–2019 and 2020–2025).
2. **Preprocessing** — Tweet normalization using BERTweet's preprocessing pipeline (emoji demojization, tokenization via NLTK TweetTokenizer).
3. **Embedding Extraction** — Contextual word embeddings extracted using [BERTweet](https://github.com/VinAIResearch/BERTweet) (`vinai/bertweet-base`).
4. **Semantic Analysis**:
   - Cosine similarity & Canberra distance between time-period embeddings
   - TF-IDF analysis with SVD decomposition
   - Logistic Regression for temporal classification
   - KMeans clustering with t-SNE visualization
   - Dependency parsing & POS tagging via spaCy
   - Sentence-level semantic comparison using Sentence-BERT (`all-MiniLM-L6-v2`)
5. **Statistical Testing** — Fisher's exact test, Chi-square test, and Bonferroni correction via `statsmodels`.

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | BERTweet (`vinai/bertweet-base`) |
| Sentence Embeddings | Sentence-BERT (`all-MiniLM-L6-v2`) |
| NLP Pipeline | spaCy (`en_core_web_sm`) |
| ML Framework | PyTorch, scikit-learn |
| Statistical Analysis | SciPy, statsmodels |
| Preprocessing | NLTK, emoji |

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/satriyoaria/BERTweet-SemanticShift-FromMemeToMainstream.git
cd BERTweet-SemanticShift-FromMemeToMainstream

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

> **Note:** This notebook was originally developed on **Google Colab** with GPU runtime. For local execution, a CUDA-compatible GPU is recommended for faster BERTweet inference.

## Usage

Open the Jupyter notebook:

```bash
jupyter notebook script/BERTweet-SemanticShifting.ipynb
```

Or upload it to [Google Colab](https://colab.research.google.com/) for GPU-accelerated execution.

## License

This project is part of an academic thesis. Please cite appropriately if used for research purposes.
