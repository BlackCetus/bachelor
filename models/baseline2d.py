import torch
import torch.nn as nn
import torch.nn.functional as F
#import data.data as d


class baseline2d(nn.Module):
    def __init__(self, embedding_size, output_size):
        super(baseline2d, self).__init__()

        self.conv = nn.Conv2d(embedding_size, output_size, kernel_size=(1, 1))

        self.fc1 = nn.Linear(output_size, output_size)
        self.fc2 = nn.Linear(output_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, protein1, protein2):

        proteins = torch.einsum('ik,jk->ijk', protein1, protein2)
        proteins = proteins.permute(2, 0, 1)
        x = self.conv(proteins.unsqueeze(0))

        x = x.max(dim=2)[0]

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        x = self.sigmoid(x)

        return x.squeeze()


model = baseline2d(embedding_size=512, output_size=256)

protein1 = torch.randn(50, 512)  # An example protein embedding
protein2 = torch.randn(80, 512)  # Another example protein embedding
output = model(protein1, protein2)    