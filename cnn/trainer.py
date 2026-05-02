# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
from debug import *
import debug
import cnn_models
import make_data
from seeder import *
import torch
import numpy as np
import common
from torch.utils.data import DataLoader
from torch import nn
from argParser import args


set_seed(args.seed)
set_verbosity(args.verbosity)
cnn = cnn_models.CNN(args.hidden_size).to(common.device)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(params =cnn.parameters(), lr=args.lr)

d = make_data.DataInput(num_samples=args.num_samples)
train_size = int(len(d) * 0.8)
val_size = int(len(d)) - train_size

train_set, val_set = torch.utils.data.random_split(d, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle = True)
val_loader = DataLoader(val_set, batch_size=args.batch_size)

for i in range(0, args.epochs):
    train_loss = 0
    train_correct = 0
    
    for inputs, labels in train_loader:
        dInputs, dLabels = inputs.to(common.device), labels.to(common.device)
        optimizer.zero_grad()
        output = cnn.forward( dInputs ) 
        preds = output.argmax(dim=1)
        train_correct += (preds == dLabels).sum().item()

        loss = loss_fn (output, dLabels)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
    accuracy = train_correct / len(train_set)
    train_loss /= len(train_loader)
   
    debug.output( f"Epoch: {i}, Training Loss: {train_loss}, Accuracy: {accuracy}")

    with torch.no_grad():
        val_loss = 0
        correct = 0
        displayed = False
        for inputs, labels in val_loader:
            dInputs, dLabels = inputs.to(common.device), labels.to(common.device)

            output = cnn.forward(dInputs)
            val_loss += loss_fn(output, dLabels).item()
            preds = output.argmax(dim=1)
            correct += (preds == dLabels).sum().item()
            
            if (not displayed and debug.checkVerbosity(INFO)):
                probabilities = output.exp()
                text = []
                for loop in range(len(preds)):
                    currentText = ""
                    if (preds[loop] != dLabels[loop]):
                        currentText = " ERROR"
                    else:  currentText = " OK"
                    text.append( currentText + f" Circle: {probabilities[loop][0]:.2f} Rect: {probabilities[loop][1]:.2f}")

                if (i == args.epochs - 1):
                    make_data.display_images( dInputs.squeeze(1).permute(1,2,0).cpu() , text)
                displayed = True

        val_loss /= len(val_loader)        
        accuracy = correct / (len(val_set)) 
        debug.output( f"Epoch: {i}, Validation Loss: {val_loss}, Accuracy: {accuracy}")

