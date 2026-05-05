# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
from nltk.tokenize import word_tokenize
import gensim.downloader as api

wv = api.load("glove-wiki-gigaword-100")

text = "the cat sat on the mat"
tokens = word_tokenize(text.lower())
# ['the', 'cat', 'sat', 'on', 'the', 'mat']

vectors = []
for token in tokens:
    if token in wv:
        vectors.append(wv[token])      # shape (100,)
    else:
        vectors.append(np.zeros(100))  # OOV → zero vector

sequence = np.stack(vectors)           # shape (seq_len, 100)