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



set_seed(73)
cnn = cnn_models.CNN().to(common.device)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(params =cnn.parameters(), lr=0.1)

d = make_data.DataInput()
train_size = int(len(d) * 0.8)
val_size = int(len(d)) - train_size

train_set, val_set = torch.utils.data.random_split(d, [train_size, val_size])

# TBD: Get the batch_size from argParser and remove this comment after
train_loader = DataLoader(train_set, batch_size=100, shuffle = True)
val_loader = DataLoader(val_set, batch_size=100 )

#TBD: get epoch from argparser and then remove this comment
for i in range(0, 1000):
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
        for inputs, labels in val_loader:
            dInputs, dLabels = inputs.to(common.device), labels.to(common.device)

            output = cnn.forward(dInputs)
            val_loss += loss_fn(output, dLabels).item()
            preds = output.argmax(dim=1)
            correct += (preds == dLabels).sum().item()

        val_loss /= len(val_loader)        
        accuracy = correct / (len(val_set)) 
        debug.output( f"Epoch: {i}, Validation Loss: {val_loss}, Accuracy: {accuracy}")

