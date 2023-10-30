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



def metrics(y_true, y_pred):
    true_positive = torch.sum((y_true == 1) & (y_pred == 1)).item()
    pred_positive = torch.sum(y_pred == 1).item()
    real_positive = torch.sum(y_true == 1).item()
    true_negative = torch.sum((y_true == 0) & (y_pred == 0)).item()

    if real_positive == 0:
        precision = recall = f1_score = accuracy = 0
    else:
        precision = true_positive / pred_positive if pred_positive > 0 else 0
        recall = true_positive / real_positive if real_positive > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positive + true_negative) / len(y_true)

    return accuracy, precision, recall, f1_score


train_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/pan_train_all_seq_1166.csv"
test_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/pan_test_all_seq_1166.csv"
learning_rate = 0.001
num_epochs = 25
bs = 512
max = 1166

wandb.init(project="bachelor",
           config={"learning_rate": learning_rate,
                   "epochs": num_epochs,
                   "batchsize": bs,
                   "dataset": "pan"})

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
    epoch_acc = 0.0
    epoch_prec = 0.0
    epoch_rec = 0.0
    epoch_f1 = 0.0
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
        epoch_loss += loss.item()
        predicted_labels = torch.round(outputs.float())
        met = metrics(labels.to(1), predicted_labels)
        epoch_acc += met[0]
        epoch_prec += met[1]
        epoch_rec  += met[2]
        epoch_f1  += met[3]
        wandb.log({
            "batch_loss": loss,
            "batch_accuracy": met[0],
            "batch_precision": met[1],
            "batch_recall": met[2],
            "batch_f1_score": met[3],
        })
    avg_acc = epoch_acc / len(dataloader)
    avg_prec = epoch_prec / len(dataloader)
    avg_rec = epoch_rec / len(dataloader)
    avg_f1 = epoch_f1 / len(dataloader)
    avg_loss = epoch_loss / len(dataloader)
    wandb.log({
        "epoch_loss": avg_loss,
        "epoch_accuracy": avg_acc,
        "epoch_precision": avg_prec,
        "epoch_recall": avg_rec,
        "epoch_f1_score": avg_f1
    })
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}, Accuracy: {avg_acc}, Precision: {avg_prec}, Recall: {avg_rec}, F1 Score: {avg_f1}")
    
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_prec = 0.0
    val_rec = 0.0
    val_f1 = 0.0
    with torch.no_grad():
        for val_batch in vdataloader:
            val_inputs = val_batch['tensor']
            val_labels = val_batch['interaction']
            val_labels = val_labels.unsqueeze(1).float()
            val_outputs = model(val_inputs.to(1))
            predicted_labels = torch.round(val_outputs.float())
            met = metrics(val_labels.to(1), predicted_labels)
            val_acc += met[0]
            val_prec += met[1]
            val_rec  += met[2]
            val_f1  += met[3]
            val_loss += criterion(val_outputs, val_labels.to(1))

    batch_avg_loss = val_loss / len(vdataloader)
    avg_acc = val_acc / len(vdataloader)
    avg_prec = val_prec / len(vdataloader)
    avg_rec = val_rec / len(vdataloader)
    avg_f1 = val_f1 / len(vdataloader)
    wandb.log({
        "val_loss": batch_avg_loss,
        "val_accuracy": avg_acc,
        "val_precision": avg_prec,
        "val_recall": avg_rec,
        "val_f1_score": avg_f1
    })      
    print(f"Epoch {epoch+1}/{num_epochs}, Average Val Loss: {batch_avg_loss}, Val Accuracy: {avg_acc}, Val Precision: {avg_prec}, Val Recall: {avg_rec}, Val F1 Score: {avg_f1}")
torch.save(model.state_dict(), '/nfs/home/students/t.reim/bachelor/pytorchtest/models/first_model_pan.pt')
