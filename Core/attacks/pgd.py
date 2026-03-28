import torch
import torch.nn as nn

def pgd_attack(model, X, y, epsilon=0.031, step_size=0.003, num_steps=20):
    model.eval()

    X_adv = X.clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        outputs = model(X_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)

        model.zero_grad()
        if X_adv.grad is not None:
            X_adv.grad.zero_()

        loss.backward()

        eta = step_size * X_adv.grad.sign()
        X_adv = X_adv.detach() + eta

        eta = torch.clamp(X_adv - X, -epsilon, epsilon)
        X_adv = torch.clamp(X + eta, 0, 1).detach().requires_grad_(True)

    return X_adv
