import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# ===========================================================================
# Interfaz pública genérica (registrada en Metodo/__init__.py)
# ===========================================================================

def make_state(cfg, device):
    """
    D-TRADES no requiere estado persistente entre batches.
    Retorna None — train.py acepta method_state=None sin cambios.
    """
    return None


def save_state(state, path: str) -> None:
    """Sin estado persistente no hay nada que guardar."""
    pass


def load_state(state, path: str, device) -> None:
    """Sin estado persistente no hay nada que restaurar."""
    pass


def compute_loss(model, x, y, cfg, method_state):
    """
    Interfaz genérica de pérdida para train.py.

    Parámetros
    ----------
    model        : nn.Module.
    x            : batch de imágenes limpias [B, C, H, W].
    y            : etiquetas verdaderas [B].
    cfg          : instancia de Config (solo lectura).
    method_state : None (no se usa).

    Retorna
    -------
    loss : Tensor escalar con gradiente listo para .backward().
    info : dict con métricas de diagnóstico (sin gradiente).
    """
    loss_total, lam, loss_natural, loss_robust = d_trades_loss(
        model         = model,
        x_natural     = x,
        y             = y,
        step_size     = cfg.step_size,
        epsilon       = cfg.epsilon,
        perturb_steps = cfg.num_steps,
        alpha         = cfg.alpha_base,   # peso de H(x) dentro del término adversarial
        beta          = cfg.beta_base,    # amplificador de KL según sensibilidad
        gamma         = cfg.gamma,        # peso de (1 - f(x)_y) como ponderador global
    )
    info = {
        "lambda_min":   lam.min().item(),
        "lambda_max":   lam.max().item(),
        "lambda_mean":  lam.mean().item(),
        "loss_natural": loss_natural.item(),
        "loss_robust":  loss_robust.item(),
    }
    return loss_total, info


# ===========================================================================
# Normalización
# ===========================================================================

@torch.no_grad()
def _normalize_batch(v, eps=1e-8):
    """
    Normaliza un vector [B] al rango [0, 1] con min-max sobre el batch.
    eps=1e-8 evita división por cero cuando todos los valores son iguales.
    """
    vmin = v.min()
    vmax = v.max()
    return (v - vmin) / (vmax - vmin + eps)


# ===========================================================================
# Función de pérdida D-TRADES — nueva formulación
# ===========================================================================

def d_trades_loss(
    model,
    x_natural,
    y,
    step_size,
    epsilon,
    perturb_steps,
    distance='l_inf',
    alpha=0.5,
    beta=0.5,
    gamma=0.5,
    normalize_sensitivity=True,
    per_sample_sensitivity=False,
    EPS=1e-12,
):
    """
    Pérdida D-TRADES con formulación reestructurada.

    Estructura de pérdida:
    ──────────────────────
        L = L_CE(f(x), y)
          + (1 - f(x)_y) * [ lambda_S(x) * KL(f(x) || f(x')) + alpha * H(x) ]

    Donde:
        (1 - f(x)_y)        Ponderador global por muestra. Inspirado en MART
                            (Wang et al., 2020). Si el modelo ya clasifica bien
                            la muestra (p_y alto), este factor es bajo y la
                            muestra recibe poca penalización adversarial. Si el
                            modelo está confundido (p_y bajo), el factor es alto
                            y la muestra recibe máxima penalización.
                            Rango: [0, 1]. Calculado sobre la muestra LIMPIA.

        lambda_S(x)         Amplificador del KL según sensibilidad adversarial.
                            = 1 + beta * S_tilde(x)
                            Garantiza piso en 1 (el KL nunca desaparece) y
                            amplifica hasta (1 + beta) en zonas muy sensibles.
                            S_tilde(x): sensibilidad normalizada a [0, 1].
                            Rango: [1, 1 + beta].

        KL(f(x) || f(x'))   Divergencia KL entre predicción limpia y adversarial.
                            Mide cuánto cambia el modelo bajo ataque.

        alpha * H(x)        Penalización por incertidumbre en la predicción limpia.
                            H(x) y KL son ambas divergencias/entropías sobre
                            distribuciones de probabilidad — mismo orden de magnitud,
                            suma coherente dentro del corchete.
                            Rango de H(x) normalizado: [0, 1].

    Por qué esta estructura es coherente:
    ──────────────────────────────────────
        - (1 - f(x)_y) opera como peso global: escala TODA la parte adversarial
          según la dificultad de la muestra. Esto es análogo a MART pero
          manteniendo la estructura KL de TRADES.

        - lambda_S(x) opera dentro del corchete sobre el KL únicamente. La
          sensibilidad (norma del gradiente KL) tiene escala diferente a H(x),
          así que no se suma directamente sino que modula el término con el que
          comparte naturaleza (el KL).

        - alpha * H(x) y KL son compatibles en escala: ambos son no-negativos
          y del mismo orden de magnitud cuando las distribuciones son similares.
          Se suman dentro del corchete de forma coherente.

    Parámetros
    ──────────
    model                  CNN a entrenar.
    x_natural              Imágenes limpias [B, C, H, W].
    y                      Etiquetas verdaderas [B].
    step_size              Paso del ataque PGD.
    epsilon                Radio máximo de perturbación.
    perturb_steps          Pasos del ataque PGD.
    distance               Norma del ataque: 'l_inf' o 'l_2'.
    alpha                  Peso de H(x) dentro del término adversarial.
    beta                   Factor de amplificación del KL por sensibilidad.
                           beta=0 equivale a TRADES puro con ponderador MART.
    gamma                  No se usa directamente en la pérdida — (1-f(x)_y)
                           ya es el ponderador. Reservado para ablaciones futuras
                           (p.ej. gamma * (1 - f(x)_y) si se quiere escalar).
    normalize_sensitivity  Si True, normaliza S(x) a [0, 1] antes de calcular
                           lambda_S. Necesario para que beta tenga un efecto
                           consistente independientemente de la escala del gradiente.
    per_sample_sensitivity Si True, sensibilidad exacta por muestra (loop).
                           Si False, aproximación por batch (recomendado).
    EPS                    Estabilidad numérica para log(0) en entropía.

    Retorna
    ───────
    loss_total   Pérdida total (escalar, con gradiente).
    lam          Ponderador efectivo por muestra [B] = (1-f(x)_y) * lambda_S(x).
                 Detached. Útil para monitorear cuánto peso recibe cada muestra.
    loss_natural Pérdida de clasificación (detached, para logging).
    loss_robust  Término adversarial completo (detached, para logging).
    """

    # ---------- Ataque PGD ----------
    criterion_kl_sum = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = x_natural.size(0)

    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                loss_kl = criterion_kl_sum(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1)
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif distance == 'l_2':
        delta = 0.001 * torch.randn_like(x_natural).detach()
        delta = Variable(delta.data, requires_grad=True)
        opt_delta = torch.optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            opt_delta.zero_grad()
            with torch.enable_grad():
                loss = -criterion_kl_sum(
                    F.log_softmax(model(adv), dim=1),
                    F.softmax(model(x_natural), dim=1)
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
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)

        x_adv = Variable(x_natural + delta, requires_grad=False)

    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    # ---------- Forward limpio y adversarial ----------
    logits_nat    = model(x_natural)
    logits_adv    = model(x_adv.detach())

    probs_nat     = F.softmax(logits_nat, dim=1).clamp(min=1e-8, max=1.0)
    log_probs_adv = F.log_softmax(logits_adv, dim=1).clamp(min=-50, max=0)

    # Pérdida de clasificación sobre muestra limpia
    loss_natural = F.cross_entropy(logits_nat, y, reduction='mean')

    # KL por muestra [B]: KL(f(x) || f(x'))
    kl_per_example = F.kl_div(
        log_probs_adv, probs_nat, reduction='none'
    ).sum(dim=1)

    # ---------- H(x): entropía local [B] ----------
    # H(x) = -sum_c p_c * log(p_c)
    # Alta cuando el modelo está confundido (distribución plana).
    # Baja cuando el modelo está seguro (distribución concentrada).
    # Mismo orden de magnitud que KL — coherente como término aditivo.
    entropy = -(probs_nat * torch.log(probs_nat.clamp_min(EPS))).sum(dim=1)

    # Normaliza H(x) al rango [0, 1] para que alpha tenga escala consistente.
    entropy_n = _normalize_batch(entropy)

    # ---------- S(x): sensibilidad adversarial [B] ----------
    # S(x) = || nabla_{x'} KL(f(x) || f(x')) ||_2
    # Mide cuánto varía la KL ante pequeñas perturbaciones de x'.
    # No se suma directamente a la pérdida — modula el KL dentro de lambda_S.
    if per_sample_sensitivity:
        # Exacta por muestra: backwards individuales. Útil para análisis.
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
        # Aproximación por batch: un único backward sobre KL promedio.
        # Suficiente para modular el KL — recomendado para entrenamiento.
        x_adv_req = x_adv.detach().clone().requires_grad_(True)
        kl_total  = F.kl_div(
            F.log_softmax(model(x_adv_req), dim=1),
            probs_nat.detach(),
            reduction='batchmean'
        )
        g = torch.autograd.grad(kl_total, x_adv_req, create_graph=False)[0]
        sensitivity = g.view(batch_size, -1).norm(p=2, dim=1)   # [B]

    # Normaliza S(x) a [0, 1] para que beta tenga un efecto consistente
    # independientemente de la magnitud absoluta del gradiente.
    sensitivity_n = _normalize_batch(sensitivity) if normalize_sensitivity else sensitivity

    # ---------- lambda_S(x): amplificador del KL por sensibilidad [B] ----------
    # lambda_S(x) = 1 + beta * S_tilde(x)
    # Piso en 1: el KL nunca desaparece aunque la sensibilidad sea mínima.
    # Techo en (1 + beta): la amplificación máxima es controlada por beta.
    # Se detacha porque es un peso, no un término que el modelo deba optimizar.
    lambda_S = (1.0 + beta * sensitivity_n).detach()   # [B], rango [1, 1+beta]

    # ---------- (1 - f(x)_y): ponderador global por muestra [B] ----------
    # Probabilidad de error en la clase correcta, calculada sobre la muestra LIMPIA.
    # Si p_y = 0.9 → peso = 0.1 (muestra fácil, poca penalización adversarial).
    # Si p_y = 0.3 → peso = 0.7 (muestra difícil, mucha penalización adversarial).
    # Rango natural [0, 1]. No requiere normalización.
    # Se detacha: es un peso, no un término a optimizar directamente.
    error_weight = (1.0 - probs_nat[torch.arange(batch_size), y]).detach()   # [B]

    # ---------- Término adversarial reestructurado ----------
    # Dentro del corchete:
    #   lambda_S(x) * KL  →  KL amplificado por sensibilidad local
    #   alpha * H(x)      →  penalización por incertidumbre (misma escala que KL)
    #
    # Todo el corchete ponderado por (1 - f(x)_y):
    #   muestras difíciles (p_y bajo)  → peso alto → más penalización adversarial
    #   muestras fáciles  (p_y alto)   → peso bajo → menos penalización adversarial
    inner_term   = lambda_S * kl_per_example + alpha * entropy_n   # [B]
    loss_robust_dynamic = (error_weight * inner_term).mean()       # escalar

    # lam: ponderador efectivo por muestra para logging.
    # Captura cuánto peso total recibe la parte adversarial de cada muestra.
    # = (1 - f(x)_y) * lambda_S(x), el factor multiplicativo del KL.
    lam = (error_weight * lambda_S).detach()   # [B]

    # ---------- Pérdida total ----------
    # L = L_CE(f(x), y) + (1 - f(x)_y) * [lambda_S(x) * KL + alpha * H(x)]
    loss_total = loss_natural + loss_robust_dynamic

    return loss_total, lam, loss_natural.detach(), loss_robust_dynamic.detach()
