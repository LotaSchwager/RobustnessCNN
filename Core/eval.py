# core/eval.py
import torch
from core.attacks import pgd_linf_whitebox

@torch.no_grad()
def _count_errors(model, x, y):
    pred = model(x).argmax(1)
    return (pred != y).float().sum().item()

def eval_adv_test_whitebox_pgd(
    model,
    device,
    test_loader,
    epsilon: float,
    num_steps: int,
    step_size: float,
):
    model.eval()

    natural_err_total = 0.0
    robust_err_total = 0.0
    n = len(test_loader.dataset)

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        # Natural
        natural_err_total += _count_errors(model, x, y)

        # Robust (PGD)
        x_adv = pgd_linf_whitebox(
            model, x, y,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            clamp=(0.0, 1.0)
        )
        robust_err_total += _count_errors(model, x_adv, y)

    natural_acc = 1.0 - natural_err_total / n
    robust_acc  = 1.0 - robust_err_total / n
    robust_drop = natural_acc - robust_acc
    attack_success_rate = 1.0 - robust_acc  # = robust error rate

    return {
        "natural_acc": float(natural_acc),
        "robust_acc": float(robust_acc),
        "robust_drop": float(robust_drop),
        "attack_success_rate": float(attack_success_rate),
    }