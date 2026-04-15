import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DTRADES(nn.Module):
    def __init__(self, step_size=0.003, epsilon=0.031, perturb_steps=10, 
                 distance='l_inf', alpha_base=1.0, beta_base=1.0, 
                 per_sample_sensitivity=False, EPS=1e-12):
        """
        D-TRADES original implementation.
        
        Parámetros:
          step_size      -> Paso del ataque PGD.
          epsilon        -> Radio máximo de perturbación adversarial.
          perturb_steps  -> Número de pasos del ataque PGD.
          distance       -> Norma del ataque: 'l_inf' o 'l_2'.
          alpha_base     -> Peso de H_n(x) fijo para todas las clases.
          beta_base      -> Peso de S_n(x) fijo para todas las clases.
          per_sample_sensitivity -> Si True, sensibilidad exacta por muestra (más lento).
          EPS            -> Constante para estabilidad numérica.
        """
        super(DTRADES, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.alpha_base = alpha_base
        self.beta_base = beta_base
        self.per_sample_sensitivity = per_sample_sensitivity
        self.EPS = EPS
        self.criterion_kl_sum = nn.KLDivLoss(reduction='sum')

    def forward(self, model, x_natural, y):
        # ---------- Ataque PGD ----------
        model.eval()
        batch_size = x_natural.size(0)

        x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()

        # Pre-calculamos probabilidades limpias una sola vez para el ataque
        with torch.no_grad():
            probs_nat_pgd = F.softmax(model(x_natural), dim=1)

        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_(True)
                with torch.enable_grad():
                    loss_kl = self.criterion_kl_sum(
                        F.log_softmax(model(x_adv), dim=1),
                        probs_nat_pgd
                    )
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                
        elif self.distance == 'l_2':
            delta = 0.001 * torch.randn_like(x_natural).detach()
            delta = Variable(delta.data, requires_grad=True)
            opt_delta = torch.optim.SGD([delta], lr=self.epsilon / self.perturb_steps * 2)

            for _ in range(self.perturb_steps):
                adv = x_natural + delta
                opt_delta.zero_grad()
                with torch.enable_grad():
                    loss = -self.criterion_kl_sum(
                        F.log_softmax(model(adv), dim=1),
                        probs_nat_pgd
                    )
                loss.backward()
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                safe = grad_norms.clone()
                safe[safe == 0] = 1.0
                delta.grad.div_(safe.view(-1, 1, 1, 1))
                zero_mask = (grad_norms == 0).view(-1, 1, 1, 1)
                delta.grad[zero_mask] = torch.randn_like(delta.grad[zero_mask])
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                opt_delta.step()
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)

            x_adv = Variable(x_natural + delta, requires_grad=False)
            
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        model.train()

        # ---------- Forward limpio y adversarial ----------
        logits_nat = model(x_natural)
        logits_adv = model(x_adv.detach())

        probs_nat     = F.softmax(logits_nat, dim=1).clamp(min=1e-8, max=1.0)
        probs_adv     = F.softmax(logits_adv, dim=1).clamp(min=1e-8, max=1.0)
        log_probs_adv = F.log_softmax(logits_adv, dim=1).clamp(min=-50, max=0)

        # Pérdida de clasificación estándar sobre datos limpios
        loss_natural = F.cross_entropy(logits_nat, y, reduction='mean')

        # KL por muestra: KL(f(x) || f(x+delta)), forma [B]
        kl_per_example = F.kl_div(
            log_probs_adv, probs_nat, reduction='none'
        ).sum(dim=1)

        # ---------- Entropía local H(x) [B] ----------
        # Normalización min-max por batch
        entropy = -(probs_nat * torch.log(probs_nat.clamp_min(self.EPS))).sum(dim=1)
        entropy_n = (entropy - entropy.min()) / (entropy.max() - entropy.min() + self.EPS)

        # ---------- Sensibilidad adversarial S(x) [B] ----------
        if self.per_sample_sensitivity:
            sens_list = []
            model.eval()
            for i in range(batch_size):
                xi = x_adv[i:i+1].detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    kl_i = F.kl_div(
                        F.log_softmax(model(xi), dim=1),
                        probs_nat[i:i+1].detach(),
                        reduction='sum'
                    )
                gi = torch.autograd.grad(kl_i, xi, create_graph=False, retain_graph=False)[0]
                sens_list.append(gi.view(gi.size(0), -1).norm(p=2, dim=1))
            sensitivity = torch.cat(sens_list, dim=0)   # [B]
            model.train()
        else:
            x_adv_req = x_adv.detach().clone().requires_grad_(True)
            kl_total = F.kl_div(
                F.log_softmax(model(x_adv_req), dim=1),
                probs_nat.detach(),
                reduction='sum'
            )
            g = torch.autograd.grad(kl_total, x_adv_req, create_graph=False)[0]
            sensitivity = g.view(batch_size, -1).norm(p=2, dim=1)   # [B]

        # Normalización min-max por batch
        sensitivity_n = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min() + self.EPS)

        # ---------- Construcción de lambda [B] ----------
        # Para la versión original, usamos valores fijos base independientemente de la clase, sin softplus ni validacion de error (gamma)
        lam = (self.alpha_base * entropy_n + self.beta_base * sensitivity_n).detach()

        # ---------- Pérdida robusta ponderada ----------
        loss_robust_dynamic = (lam * kl_per_example).mean()

        # L = L_CE(f(x), y) + L_robust
        loss_total = loss_natural + loss_robust_dynamic

        # ---------- INFO ----------
        pred = logits_adv.argmax(dim=1).detach()
        lam_np = lam.detach().cpu().numpy()
        zeros = np.zeros_like(lam_np)

        info = {
            "lam":          lam_np,
            "lam_raw":      zeros,
            "entropy":      entropy_n.detach().cpu().numpy(),
            "sensitivity":  sensitivity_n.detach().cpu().numpy(),
            "error":        zeros,
            "alpha_per_class": zeros,
            "beta_per_class": zeros,
            "predictions":  pred.cpu().numpy(),
            "targets":      y.detach().cpu().numpy(),
            "loss_natural": loss_natural.item(),
            "loss_robust":  loss_robust_dynamic.item(),
        }

        return loss_total, info


# ===========================================================================
# Interfaz pública genérica (registrada en Metodo/__init__.py)
# ===========================================================================

def make_state(cfg, device):
    """
    Retorna la instancia de nuestra clase DTRADES
    """
    distance = getattr(cfg, 'distance', 'l_inf')
    per_sample_sens = getattr(cfg, 'per_sample_sensitivity', False)
    
    criterion = DTRADES(
        step_size=cfg.step_size,
        epsilon=cfg.epsilon,
        perturb_steps=cfg.num_steps,
        distance=distance,
        alpha_base=getattr(cfg, 'alpha_base', 1.0),
        beta_base=getattr(cfg, 'beta_base', 1.0),
        per_sample_sensitivity=per_sample_sens
    ).to(device)
    return criterion

def save_state(state, path: str) -> None:
    pass

def load_state(state, path: str, device) -> None:
    pass

def compute_loss(model, x, y, cfg, method_state):
    """
    Interfaz genérica de pérdida para train.py.
    """
    criterion = method_state
    loss_total, result = criterion(model, x, y)
    return loss_total, result