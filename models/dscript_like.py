import torch
import torch.nn as nn
import torch.nn.functional as F
import data.data as d


class DScriptLike(nn.Module):

    def __init__(self, insize, d=100, w=7, h=50,
                x0 = 0.5, k = 20,
                do_pool = False, pool_size=9, do_w = True, theta_init=1, lambda_init=0, gamma_init = 0):
        
        super(DScriptLike, self).__init__()
        self.insize = insize
        # activation func params
        self.k = nn.Parameter(torch.FloatTensor([float(k)]), requires_grad=True)
        self.x0 = x0

        # interaction module params
        self.do_w = do_w
        self.do_pool = do_pool
        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)
        # for weighing contact map
        self.xx = nn.Parameter(torch.arange(2000), requires_grad=False)

        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        if self.do_w:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))

        self.clip()    

        # == FullyConnectedEmbed = embedding
        self.fc1 = nn.Linear(insize, d)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        # from contact.py: FullyConnected
        self.conv2 = nn.Conv2d(2 * d, h, 1)
        self.relu2 = nn.ReLU()          
        self.batchnorm1 = nn.BatchNorm2d(h)

        #from contact.py: ContactCNN
        self.conv = nn.Conv2d(h, 1, w, padding=w // 2)
        self.batchnorm2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU()
        
   
    def forward(self, x1, x2):

        # == FullyConnectedEmbed = embedding
        x1 = x1.to(torch.float32).unsqueeze(0)
        x2 = x2.to(torch.float32).unsqueeze(0)

        x1 = x1.contiguous()
        x1 = x1.view(x1.size(0),-1, self.insize)
        x1 = self.fc1(x1)

        x2 = x2.contiguous()
        x2 = x2.view(x2.size(0),-1, self.insize)
        x2 = self.fc1(x2)

        # from contact.py: FullyConnected
        diff = torch.abs(x1.unsqueeze(2) - x2.unsqueeze(1))
        mul = x1.unsqueeze(2) * x2.unsqueeze(1)

        m = torch.cat([diff, mul], dim=-1)

        m = m.permute(0, 3, 1, 2)
        m = self.conv2(m)
        m = self.batchnorm1(m)
        m = self.relu2(m)


        #from contact.py: ContactCNN
        C= self.conv(m)
        C = self.batchnorm2(C)
        C = self.relu2(C)

        # from interaction.py: map_predict
        if self.do_w:
            N, M = C.shape[2:]

            x1 = -1 * torch.square(
                (self.xx[:N] + 1 - ((N + 1) / 2)) / (-1 * ((N + 1) / 2))
            )

            x2 = -1 * torch.square(
                (self.xx[:M] + 1 - ((M + 1) / 2)) / (-1 * ((M + 1) / 2))
            )

            x1 = torch.exp(self.lambda_ * x1)
            x2 = torch.exp(self.lambda_ * x2)

            W = x1.unsqueeze(1) * x2
            W = (1 - self.theta) * W + self.theta
            yhat = C * W

        else:
            yhat = C
        if self.do_pool:
            yhat = self.maxPool(yhat)    

        if True:
            mu = torch.mean(yhat)
            sigma = torch.var(yhat)
            Q = torch.relu(yhat - mu - (self.gamma * sigma))
        else:   
            mean = torch.mean(yhat, dim=[1,2], keepdim=True)
            std_dev = torch.sqrt(torch.var(yhat, dim=[1,2], keepdim=True) + 1e-5)
            Q = torch.relu(yhat - mean - (self.gamma * std_dev))

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
        return torch.stack(pred)      

    def clip(self):
        """
        Clamp model values

        :meta private:
        """
        if self.do_w:
            self.theta.data.clamp_(min=0, max=1)
            self.lambda_.data.clamp_(min=0)

        self.gamma.data.clamp_(min=0)



        
        
        
        
        
