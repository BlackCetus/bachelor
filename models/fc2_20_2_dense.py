import torch
import torch.nn as nn
import torch.nn.functional as F


class FC2_20_2Dense(nn.Module):

    def __init__(self, insize, mean, max_len):
        super(FC2_20_2Dense, self).__init__()
        
        # Define layers and modules
        self.fc1 = nn.Linear(insize, 20)
        self.fc2 = nn.Linear(20, 20)

        if not mean:
            self.bn1 = nn.BatchNorm2d(1)
            self.bn2 = nn.BatchNorm2d(1)
            self.bn3 = nn.BatchNorm2d(1)
            self.bn4 = nn.BatchNorm2d(1)
            self.bn5 = nn.BatchNorm2d(1)
            self.fc7 = nn.Linear(max_len, 1)
        else:
            self.bn1 = nn.BatchNorm1d(20)
            self.bn2 = nn.BatchNorm1d(20)
            self.bn3 = nn.BatchNorm1d(20)
            self.bn4 = nn.BatchNorm1d(20)
            self.bn5 = nn.BatchNorm1d(20)

        # The same is done for input sequence 2.
        self.fc3 = nn.Linear(insize, 20)
        self.fc4 = nn.Linear(20, 20)

        # Both outputs are concatenated and fed to a fully connected layer with 20 neurons. Then, batch normalization is applied.
        self.fc5 = nn.Linear(40, 20)
        # The output of this layer is fed to a fully connected layer with 1 neuron.
        self.fc6 = nn.Linear(20, 1)

        # The model has 2 classes, 0 and 1
        self.classes = (0,1)

        self.mean = mean

    def forward(self, x):
        print(x.shape)
        test = x.split(1, dim=1)

        x1 = test[0]
        print(x1.shape)
        x1 = x1.to(torch.float32)
        x1 = x1.contiguous()
        x1 = x1.view(x.size(0),-1, 1280)
        print(x1.shape)

        x2 = test[1]
        x2 = x2.to(torch.float32)
        x2 = x2.contiguous()
        x2 = x2.view(x.size(0),-1, 1280)

        x1 = F.relu(self.fc1(x1))
        print(x1.shape)
        if not self.mean:
            x1 = x1.unsqueeze(1)  # Add an extra dimension
        x1 = self.bn1(x1)
        if not self.mean:
            x1 = x1.squeeze(1)  # Remove the extra dimension
        x1 = F.relu(self.fc2(x1))
        if not self.mean:
            x1 = x1.unsqueeze(1)
        x1 = self.bn2(x1)
        if not self.mean:
            x1 = x1.squeeze(1)  
        print(x1.shape)

        x2 = F.relu(self.fc3(x2))
        if not self.mean:
            x2 = x2.unsqueeze(1) 
        x2 = self.bn3(x2)
        if not self.mean:
            x2 = x2.squeeze(1) 
        x2 = F.relu(self.fc4(x2))
        if not self.mean:
            x2 = x2.unsqueeze(1)  
        x2 = self.bn4(x2)
        if not self.mean:
            x2 = x2.squeeze(1) 
        print(x2.shape)

        if not self.mean:
            x = torch.cat((x1,x2), 2)
        else:
            x = torch.cat((x1,x2), 1)
        x = F.relu(self.fc5(x))
        if not self.mean:
            x = x.unsqueeze(1) 
        x = self.bn5(x)
        if not self.mean:
            x = x.squeeze(1) 
        x = self.fc6(x)
        x = x.view(x.size(0), -1)
        x = self.fc7(x)
        print(x.shape)

        # classification is done using a sigmoid function
        x = torch.sigmoid(x)
        print(x.shape)
        return x
    
    def get_classes(self):
        return self.classes