import torch
import torch.nn as nn
import torch.nn.functional as F
import data.data as d


class DScriptLike(nn.Module):

    def __init__(self, insize, d=100, w=3, h=20, x0 = 0, k = 1, gamma_init = 0, pool_size=9):
        super(DScriptLike, self).__init__()
        self.k = k
        self.x0 = x0
        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        self.fc1 = nn.Linear(insize, d)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(2*d, h)
        self.relu2 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(h)

        self.conv = nn.Conv2d(h, 1, kernel_size=(2*w+1, 2*w+1), padding=w)

        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)

        self.batchnorm2 = nn.BatchNorm2d(1)
   
    def forward(self, x1, x2):

        x1 = x1.to(torch.float32).unsqueeze(0)
        x2 = x2.to(torch.float32).unsqueeze(0)

        x1 = x1.contiguous()
        x1 = x1.view(x1.size(0),-1, 1280)
        x1 = self.fc1(x1)

        x2 = x2.contiguous()
        x2 = x2.view(x2.size(0),-1, 1280)
        x2 = self.fc1(x2)


        diff = torch.abs(x1.unsqueeze(2) - x2.unsqueeze(1))
        mul = x1.unsqueeze(2) * x2.unsqueeze(1)

        m = torch.cat([diff, mul], dim=-1)

        m = m.view(-1, m.size(-1))
        m = self.fc2(m)
        m = self.batchnorm1(m)
        m = self.relu2(m)

        m = m.view(x1.size(0), x1.size(1), x2.size(1), -1)
        m = m.permute(0, 3, 1, 2)

        # Apply convolution
        C= self.conv(m)
        C = self.batchnorm2(C)
        C = torch.sigmoid(C)

        # Interaction prediction module

        B = self.maxPool(C)

        mean = torch.mean(B, dim=[1,2], keepdim=True)
        std_dev = torch.sqrt(torch.var(B, dim=[1,2], keepdim=True) + 1e-5)
        Q = torch.relu(B - mean - (self.gamma * std_dev))

        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)

        phat = torch.clamp(
            1 / (1 + torch.exp(-self.k * (phat - self.x0))), min=0, max=1
        )

        return phat, C


    def batch_iterate(self, batch, device, layer, emb_dir):
        pred = []
        for i in range(len(batch['interaction'])):
            id1 = batch['name1'][i]
            id2 = batch['name2'][i]
            seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).to(device)
            seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).to(device)
            p, cm = self.forward(seq1, seq2)
            pred.append(p)
        return torch.stack(pred).unsqueeze(1)       




        
        
        
        
        
