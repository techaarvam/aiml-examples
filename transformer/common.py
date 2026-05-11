# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

vecDims = None  # set by DataInput after loading embeddings
vocabSize = None # in the training data, all the words are sorted and enumerated and indexed. 
                 #Larger training data is likely to have a larger vocabulary.

wordDict = None # This is a look up dictionary. use as wordDict[word] to get the index. word is a token.
wordVecs = None # the token -> vector mapping loading using gensim.downloader.load(glove-wiki-gigaword-100 or similar)
dtype    = None # set in trainer.py based on args.float_type

