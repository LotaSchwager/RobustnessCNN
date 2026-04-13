import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hiperparámetros internos de MART (paper Wang et al., 2020).
# Aislados de cfg para no cruzarlos con los de D-TRADES.
MART_PARAMS = {
    "beta":          6.0,     # Peso del término KL ponderado
    "epsilon":       8/255,   # Radio máximo de perturbación
    "step_size":     2/255,   # Paso del ataque PGD
    "perturb_steps": 10,      # Número de pasos del ataque PGD
}


def mart_loss(
    model,
    x_natural,
    y,
    step_size=0.007,
    epsilon=0.031,
    perturb_steps=10,
    beta=6.0,
    distance='l_inf',
):
    """
    MART: Misclassification Aware adveRsarial Training (Wang et al., 2020).

    Implementación FIEL al repositorio oficial:
        https://github.com/YisenWang/MART/blob/master/mart.py

    No se introducen estabilizaciones numéricas ni reescalados que se desvíen
    del paper, para que la comparación con D-TRADES sea limpia.

    Pérdida total:
        L = BCE(p_adv, y)  +  beta * (1/B) * sum_i (1 - p_nat[y_i]) * KL(p_nat || p_adv)
    """
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    # ---- Ataque PGD-CE -------------------------------------------------------
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    # ---- Forward limpio y adversarial (separados, como en el paper) ----------
    # Importante: el paper hace dos forwards separados, no torch.cat. La versión
    # concatenada altera las estadísticas de BatchNorm y se aleja del método.
    logits     = model(x_natural)
    logits_adv = model(x_adv)

    # ---- Boosted Cross Entropy ----------------------------------------------
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1      = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y     = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = (
        F.cross_entropy(logits_adv, y)
        + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    )

    # ---- KL ponderado por (1 - p_nat[y]) ------------------------------------
    nat_probs  = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, y.unsqueeze(1).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1)
        * (1.0000001 - true_probs)
    )

    loss = loss_adv + float(beta) * loss_robust

    # Para que el sistema de logging muestre algo en la columna de lambda.
    # En MART beta es constante por construcción.
    lam_array = np.full(batch_size, float(beta), dtype=np.float32)

    info = {
        "lam":          lam_array,
        "loss_natural": float(loss_adv.detach().item()),
        "loss_robust":  float(loss_robust.detach().item()),
    }
    return loss, info


def compute_loss(model, x, y, cfg, method_state):
    """Interfaz genérica de pérdida para train.py."""
    loss_total, result = mart_loss(
        model         = model,
        x_natural     = x,
        y             = y,
        step_size     = MART_PARAMS["step_size"],
        epsilon       = MART_PARAMS["epsilon"],
        perturb_steps = MART_PARAMS["perturb_steps"],
        beta          = MART_PARAMS["beta"],
        distance      = 'l_inf',
    )
    return loss_total, result


def make_state(cfg, device):
    return None


def save_state(state, base_path):
    pass


def load_state(state, base_path, device):
    pass
