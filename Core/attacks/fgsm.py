import torch
import torch.nn as nn

def fgsm_attack(model, X, y, epsilon=0.031):
    model.eval()

    X_adv = X.clone().detach().requires_grad_(True)

    outputs = model(X_adv)
    loss = nn.CrossEntropyLoss()(outputs, y)

    model.zero_grad()
    loss.backward()

    eta = epsilon * X_adv.grad.sign()
    X_adv = torch.clamp(X_adv + eta, 0, 1)

    return X_adv.detach()
