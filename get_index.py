import torch

N = 2100
torch.save( torch.randperm(N), 'index2100.pt')