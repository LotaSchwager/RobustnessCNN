# core/attacks.py
import torch
import torch.nn.functional as F

def pgd_linf_whitebox(
    model,
    x,
    y,
    epsilon: float,
    num_steps = 20,
    step_size = 0.003,
    clamp=(0.0, 1.0),
):
    """
    Devuelve x_adv (no errores). Así es reutilizable por cualquier evaluator.
    """
    model.eval()

    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(*clamp)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
            x_adv = x_adv.clamp(*clamp)

        x_adv = x_adv.detach()

    return x_adv