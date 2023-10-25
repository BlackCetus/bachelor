import data.data as d
import models.first_pytorch as m
import models.fc2_20_2_dense as m2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import wandb


train_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/train_all_seq.csv"
test_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/test_all_seq.csv"
learning_rate = 0.01
num_epochs = 25
bs = 128

wandb.init(project="bachlor",
           config={"learning_rate": learning_rate,
                   "epochs": num_epochs,
                   "batchsize": bs,
                   "dataset": "Huang"})

max = max(d.max_sequence_size(train_data), d.max_sequence_size(test_data))
insize = max*24

model = m2.FC2_20_2Dense(insize=insize)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
dataset = d.MyDataset(train_data, max)
dataloader = data.DataLoader(dataset, batch_size=bs, shuffle=True)

vdataset = d.MyDataset(test_data, max)
vdataloader = data.DataLoader(vdataset, batch_size=bs, shuffle=True)

if torch.cuda.is_available():       # dont hardcode devices!
    torch.cuda.device(1)
    model.to(1)
    print('Using Cuda')


for epoch in range(num_epochs):
    epoch_loss = 0.0
    total_correct = 0
    model.train()
    for batch in dataloader:
        inputs = batch['tensor']
        labels = batch['interaction']
        labels = labels.unsqueeze(1).float()
        optimizer.zero_grad()
        outputs = model(inputs.to(1)) 
        loss = criterion(outputs, labels.to(1))
        loss.backward()
        optimizer.step()
        wandb.log({"batch_loss": loss})
        epoch_loss += loss.item()
        predicted_labels = outputs.argmax(dim=1)
        total_correct += (predicted_labels == labels.to(1)).sum().item()
    avg_loss = epoch_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    wandb.log({"epoch_loss": avg_loss, "accuracy": accuracy})
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}, Accuracy: {accuracy}")
    
    model.eval()
    val_loss = 0.0
    batch_avrg_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for val_batch in vdataloader:
            val_inputs = val_batch['tensor']
            val_labels = val_batch['interaction']
            val_labels = val_labels.unsqueeze(1).float()
            val_outputs = model(val_inputs.to(1))
            val_loss += criterion(val_outputs, val_labels.to(1))
            predicted_labels = val_outputs.argmax(dim=1)
            total_correct += (predicted_labels == val_labels.to(1)).sum().item()
    batch_avrg_loss = val_loss / len(vdataloader)
    accuracy = total_correct / len(vdataloader.dataset)
    wandb.log({"val_loss": batch_avrg_loss, "accuracy": accuracy})
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {batch_avrg_loss}, Accuracy: {accuracy}")  

torch.save(model.state_dict(), '/nfs/home/students/t.reim/bachelor/pytorchtest/models/first_model.pt')
