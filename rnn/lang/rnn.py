# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import DataInput
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

d = DataInput.DataInput()
vecDim = d.getInputSize()
rnn = langRNN.langRNN(vecDim, args.hidden_dim).to(common.device)

if args.model_file:
    rnn.load_state_dict(torch.load(args.model_file, map_location=common.device))
    dbg_output(f"Loaded model from {args.model_file}")
else:
    loss_fn = nn.MSELoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params=rnn.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(params=rnn.parameters(), lr=args.lr)

    # Lets not do the random_split for this case.
    # Lets make the next word generation interactive as a chat bot!. No validation error measurement then!

    train_loader = DataLoader(d, batch_size = args.batch_size)

    for i in range(0, args.epochs):
        total_loss = 0.0
        total_cos = 0.0
        num_batches = 0

        for inputs, labels in train_loader:
            dInputs, dLabels = inputs.to(common.device), labels.to(common.device)
            optimizer.zero_grad()

            output = rnn.forward(dInputs)

            loss = loss_fn(output, dLabels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_cos += torch.nn.functional.cosine_similarity(output.detach(), dLabels, dim=1).mean().item()
            num_batches += 1

        dbg_output(f"Epoch {i+1}: loss={total_loss/num_batches:.4f}  cosine_sim={total_cos/num_batches:.4f}")

    torch.save(rnn.state_dict(), args.input + ".model.pt")
    dbg_output(f"Model saved to {args.input}.model.pt")

wordVecs = d.getWordVecs()

print(f"Enter {args.window_size} words (space separated):")
userInput = input()
userTokens = word_tokenize(userInput.lower())

vectors = []
for token in userTokens:
    if token in wordVecs:
        vectors.append(wordVecs[token])
    else:
        vectors.append(np.zeros(100))
    if len(vectors) == args.window_size: break
while len(vectors) < args.window_size:
    vectors.append(np.zeros(100))

vectors = torch.tensor(np.array(vectors), dtype=torch.float32)
vectors = vectors.unsqueeze(0).to(common.device)

generated = list(userTokens[:args.window_size])

for i in range(0, args.output_size):
    with torch.no_grad():
        output = rnn.forward(vectors)
        word = random.choice(wordVecs.similar_by_vector(output.squeeze(0).cpu().numpy(), topn=5))[0]
        generated.append(word)
        vectors = torch.cat((vectors, output.unsqueeze(1)), dim=1)[:,1:,:]

print(" ".join(generated))

while True:
    again = input("\nPress Enter to generate more, n for new input, or q to quit: ")
    if again.strip().lower() == 'q':
        break
    elif again.strip().lower() == 'n':
        print(f"Enter {args.window_size} words (space separated):")
        userInput = input()
        userTokens = word_tokenize(userInput.lower())
        vectors = []
        for token in userTokens:
            if token in wordVecs:
                vectors.append(wordVecs[token])
            else:
                vectors.append(np.zeros(100))
            if len(vectors) == args.window_size: break
        while len(vectors) < args.window_size:
            vectors.append(np.zeros(100))
        vectors = torch.tensor(np.array(vectors), dtype=torch.float32)
        vectors = vectors.unsqueeze(0).to(common.device)
        generated = list(userTokens[:args.window_size])
    else:
        with torch.no_grad():
            output = rnn.forward(vectors)
            word = random.choice(wordVecs.similar_by_vector(output.squeeze(0).cpu().numpy(), topn=5))[0]
            generated.append(word)
            vectors = torch.cat((vectors, output.unsqueeze(1)), dim=1)[:,1:,:]
    print(" ".join(generated))


