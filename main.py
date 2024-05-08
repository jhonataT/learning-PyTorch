import torch;
from torch import nn;
from torch.utils.data import Dataset, DataLoader;
import torch.distributions.uniform as urand;
import numpy as np;

class LineNetwork(nn.Module):
    def __init__(self):
        super().__init__();
        self.layers = nn.Sequential(
            nn.Linear(1, 1)
        );

    #  Como a rede computa
    def forward(self, x):
        return self.layers(x);

class AlgebraicDataset(Dataset):
    def __init__(self, f, interval, nsamples):
        X = urand.Uniform(interval[0], interval[1]).sample([nsamples]);
        self.data = [(x, f(x)) for x in X];

    # Quantos dados o dataset possui
    def __len__(self):
        return len(self.data);

    def __getitem__(self, idx):
        return self.data[idx];
