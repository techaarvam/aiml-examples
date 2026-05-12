# Setup — Tech Aarvam Transformer

## 1. Clone the repo

```bash
git clone git@github.com:<your-repo>/transformer.git
cd transformer
```

## 2. Install dependencies

```bash
pip install nltk gensim onnxruntime datasets netron
```

> `torch`, `numpy`, and `tqdm` are pre-installed in the Vast.ai PyTorch image.
> `tomllib` is built into Python 3.11+.

## 3. Download NLTK tokenizer data

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

## 4. Download WikiText-103

```bash
cd raw_data
python download_wiki.py
cd ..
```

This writes two files:
- `raw_data/wikitext103.txt` — full dataset (~514 MB)
- `raw_data/wikitext50m.txt` — first 50 MB slice

## 5. Verify setup

```bash
python runner.py runs.toml smoke
```
