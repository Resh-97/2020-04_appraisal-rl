import torch
import torch.nn as nn
import torch.optim as optim

from .model import init_params

class AppraisalModel(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        self.appraisal = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )
        self.apply(init_params)
    
    def forward(self, embedding):
        return self.appraisal(embedding)