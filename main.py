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


def metrics( y_true, y_pred ):
    # Count positive samples.
    diff = y_true + y_pred - 1
    true_positive = sum( diff == 1 )
    pred_positive = sum( y_pred == 1 )
    real_positive = sum( y_true == 1 )

    #print('TP={}, pred pos={}, real pos={}'.format(true_positive, pred_positive, real_positive))
    
    # If there are no true samples, fix the F1 score at 0.
    if real_positive == 0:
        return 0

    # How many selected items are relevant?
    precision = true_positive / pred_positive

    # How many relevant items are selected?
    recall = true_positive / real_positive

    accuracy = (true_positive + (len(y_true) - real_positive - pred_positive)) / len(y_true)

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score


train_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/train_all_seq.csv"
test_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/test_all_seq.csv"
learning_rate = 0.01
num_epochs = 25
bs = 128

wandb.init(project="bachelor",
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
        predicted_labels = outputs.argmax(dim=1)
        met = metrics(labels, predicted_labels)
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
            predicted_labels = val_outputs.argmax(dim=1)
            met = metrics(val_labels, predicted_labels)
            val_acc += met[0]
            val_prec += met[1]
            val_rec  += met[2]
            val_f1  += met[3]
            val_loss += criterion(val_outputs, val_labels.to(1))

    batch_avrg_loss = val_loss / len(vdataloader)
    avg_acc = val_acc / len(dataloader)
    avg_prec = val_prec / len(dataloader)
    avg_rec = val_rec / len(dataloader)
    avg_f1 = val_f1 / len(dataloader)
    avg_loss = val_loss / len(dataloader)
    wandb.log({
        "val_loss": avg_loss,
        "val_accuracy": avg_acc,
        "val_precision": avg_prec,
        "val_recall": avg_rec,
        "val_f1_score": avg_f1
    })      
    print(f"Epoch {epoch+1}/{num_epochs}, Average Val Loss: {avg_loss}, Val Accuracy: {avg_acc}, Val Precision: {avg_prec}, Val Recall: {avg_rec}, Val F1 Score: {avg_f1}")
torch.save(model.state_dict(), '/nfs/home/students/t.reim/bachelor/pytorchtest/models/first_model.pt')
