import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hiperparámetros fijos internos de MART para no depender de variables globales
MART_PARAMS = {
    "beta":          6.0,     # Penalización de KL (1 - prob verdadera)
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
    distance='l_inf'
):
    """
    Implementación de MART: Misclassification Aware adveRsarial Training.
    
    A diferencia de depender de variables globales, los hiperparámetros de ataque y defensa 
    se configuran en los argumentos de esta función directamente u omitiendo cfg para que
    sea más sencillo editarlos sin cruzar con los hiperparámetros de dtrades.
    """
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    # Generar ejemplo adversarial usando PGD (paso interno necesario para MART)
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
        # l_2 no ha sido implementado, fallback a l_inf/l_2 base bounds
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
    model.train()
    x_adv = x_adv.detach()
    
    # Evaluar juntos para mantener las estadísticas de BatchNorm estables
    logits_all = model(torch.cat([x_natural, x_adv], dim=0))
    logits, logits_adv = logits_all.chunk(2, dim=0)
    
    # Probabilidades adversariales y reasignación de etiquetas para loss_adv
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    
    # Probabilidades limpias para el cálculo de divergencia
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    
    # Divergencia KL escalada por (1 - probabilidad verdadera), penalización principal
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs)
    )
    
    # Pérdida final
    loss = loss_adv + float(beta) * loss_robust
    
    # Simulamos la métrica 'lam' para mantener la coherencia de visualización en la terminal,
    # asignando beta a todo el batch de manera uniforme porque MART no es dinámico como dtrades.
    lam_array = np.full(batch_size, float(beta), dtype=np.float32)
    
    return loss, {"lam": lam_array}


def compute_loss(model, x, y, cfg, method_state):
    """
    Interfaz genérica de pérdida para train.py.
    """
    loss_total, result = mart_loss(
        model         = model,
        x_natural     = x,
        y             = y,
        step_size     = MART_PARAMS["step_size"],
        epsilon       = MART_PARAMS["epsilon"],
        perturb_steps = MART_PARAMS["perturb_steps"],
        beta          = MART_PARAMS["beta"],
        distance      = 'l_inf'
    )
    return loss_total, result

def make_state(cfg, device):
    return None

def save_state(state, base_path):
    pass

def load_state(state, base_path, device):
    pass