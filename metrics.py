import torch
import numpy as np

def psnr(y_true, y_pred, eps=1e-6):
    def standard_signal(x, dr):
        x = torch.div(x, torch.max(x))
        x = torch.clip(x, np.power(10, -dr / 10), 1)
        x = torch.log(x) / torch.log(torch.tensor(10.0)) * torch.tensor(10.0)
        return (x + dr) / (dr + eps)

    dr = 40  # dynamic range
    y_true = standard_signal(y_true, dr)
    y_pred = standard_signal(y_pred, dr)

    msef = torch.mean(torch.pow(y_pred - y_true, 2))
    maxf = torch.amax(y_true)
    psnr = 20 * torch.log10(maxf) - 10 * torch.log10(msef)
    return psnr

def nsme(y_true, y_pred,  eps=1e-6):
    return torch.sqrt( torch.mean(torch.square(y_pred-y_true))/(torch.max(y_true) - torch.min(y_true) + eps) )