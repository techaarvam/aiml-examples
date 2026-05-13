# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import DataInput
import DataInputSoftmax
import numpy as np
import torch
import langRNN
from argParser import *
from debug import *
import common
from seeder import *
from torch import nn
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
import random


set_seed(args.seed)
set_verbosity(args.verbosity)

if args.output_mode == "softmax":
    d = DataInputSoftmax.DataInputSoftmax()
    vecDim = d.getInputSize()
    vocab_size = d.getVocabSize()
    vocab = d.getVocab()
    rnn = langRNN.langRNN(vecDim, args.hidden_dim, vocab_size).to(common.device)
else:
    d = DataInput.DataInput()
    vecDim = d.getInputSize()
    rnn = langRNN.langRNN(vecDim, args.hidden_dim).to(common.device)

total_params = sum(p.numel() for p in rnn.parameters())
trainable_params = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
dbg_output(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

if args.model_file:
    rnn.load_state_dict(torch.load(args.model_file, map_location=common.device))
    dbg_output(f"Loaded model from {args.model_file}")
else:
    if args.output_mode == "softmax":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params=rnn.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(params=rnn.parameters(), lr=args.lr)

    # Lets not do the random_split for this case.
    # Lets make the next word generation interactive as a chat bot!. No validation error measurement then!

    train_loader = DataLoader(d, batch_size=args.batch_size)

    for i in range(0, args.epochs):
        total_loss = 0.0
        total_cos = 0.0
        num_batches = 0

        prev_output = None
        for inputs, labels in train_loader:
            dInputs, dLabels = inputs.to(common.device), labels.to(common.device)
            optimizer.zero_grad()

            output, _ = rnn.forward(dInputs)

            loss = loss_fn(output, dLabels)
            if args.output_mode == "glove" and args.rep_penalty > 0 and prev_output is not None:
                n = min(output.shape[0], prev_output.shape[0])
                cos_sim = torch.nn.functional.cosine_similarity(output[:n], prev_output[:n].detach(), dim=1)
                loss = loss + args.rep_penalty * cos_sim.clamp(min=0).mean()
            prev_output = output.detach()

            loss.backward()
            # Hack?!
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            if args.output_mode == "glove":
                total_cos += torch.nn.functional.cosine_similarity(output.detach(), dLabels, dim=1).mean().item()
            num_batches += 1

        if args.output_mode == "glove":
            dbg_output(f"Epoch {i+1}: loss={total_loss/num_batches:.4f}  cosine_sim={total_cos/num_batches:.4f}")
        else:
            dbg_output(f"Epoch {i+1}: loss={total_loss/num_batches:.4f}")

    torch.save(rnn.state_dict(), args.input + ".model.pt")
    dbg_output(f"Model saved to {args.input}.model.pt")


def get_input_vectors(userTokens, wordVecs, vecDim):
    vectors = []
    for token in userTokens:
        if token in wordVecs:
            vectors.append(wordVecs[token])
        else:
            vectors.append(np.zeros(vecDim))
        if len(vectors) == args.window_size: break
    while len(vectors) < args.window_size:
        vectors.append(np.zeros(vecDim))
    return torch.tensor(np.array(vectors), dtype=torch.float32).unsqueeze(0).to(common.device)


def predict_next_word(output):
    if args.output_mode == "softmax":
        probs = torch.softmax(output.squeeze(0), dim=-1)
        topk = torch.topk(probs, k=1)
        idx = random.choice(topk.indices.tolist())
        return vocab[idx], None
    else:
        word = random.choice(wordVecs.similar_by_vector(output.squeeze(0).cpu().numpy(), topn=1))[0]
        vec = torch.tensor(wordVecs[word], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(common.device)
        return word, vec


if args.output_mode == "glove":
    wordVecs = d.getWordVecs()
else:
    wordVecs = DataInput.DataInput.__new__(DataInput.DataInput)
    import gensim.downloader as api
    wordVecs = api.load("glove-wiki-gigaword-100")

print(f"Enter {args.window_size} words (space separated):")
userInput = input()
userTokens = word_tokenize(userInput.lower())
vectors = get_input_vectors(userTokens, wordVecs, vecDim)
generated = list(userTokens[:args.window_size])

h = None
for i in range(0, args.output_size):
    with torch.no_grad():
        output, h = rnn.forward(vectors, h)
        if args.output_mode == "softmax": h = None
        word, vec = predict_next_word(output)
        generated.append(word)
        if args.output_mode == "softmax":
            if word in wordVecs:
                vec = torch.tensor(wordVecs[word], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(common.device)
            else:
                vec = torch.zeros(1, 1, vecDim).to(common.device)
        vectors = torch.cat((vectors, vec), dim=1)[:,1:,:]

print(" ".join(generated))

while True:
    again = input("\nPress Enter to generate more, n for new input, or q to quit: ")
    if again.strip().lower() == 'q':
        break
    elif again.strip().lower() == 'n':
        print(f"Enter {args.window_size} words (space separated):")
        userInput = input()
        userTokens = word_tokenize(userInput.lower())
        vectors = get_input_vectors(userTokens, wordVecs, vecDim)
        generated = list(userTokens[:args.window_size])
        h = None
    else:
        for _ in range(args.output_size):
            with torch.no_grad():
                output, h = rnn.forward(vectors, h)
                if args.output_mode == "softmax": h = None
                word, vec = predict_next_word(output)
                generated.append(word)
                if args.output_mode == "softmax":
                    if word in wordVecs:
                        vec = torch.tensor(wordVecs[word], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(common.device)
                    else:
                        vec = torch.zeros(1, 1, vecDim).to(common.device)
                vectors = torch.cat((vectors, vec), dim=1)[:,1:,:]
    print(" ".join(generated))


