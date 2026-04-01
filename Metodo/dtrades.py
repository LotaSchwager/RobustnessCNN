import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# ===========================================================================
# Interfaz pública genérica (registrada en Metodo/__init__.py)
# ===========================================================================
# Todo método en Metodo/ debe exponer estas tres funciones con esta firma.
# Así train.py y main.py no necesitan saber qué método está activo.

def make_state(cfg, device):
    """
    D-TRADES no requiere estado persistente entre batches.
    Retorna None — train.py acepta method_state=None sin cambios.
    """
    return None


def save_state(state, path: str) -> None:
    """Sin estado persistente no hay nada que guardar en el checkpoint."""
    pass


def load_state(state, path: str, device) -> None:
    """Sin estado persistente no hay nada que restaurar."""
    pass


def compute_loss(model, x, y, cfg, method_state):
    """
    Interfaz genérica de pérdida para train.py.

    Parámetros
    ----------
    model        : nn.Module en modo train/eval según requiera el método.
    x            : batch de imágenes limpias [B, C, H, W].
    y            : etiquetas verdaderas [B].
    cfg          : instancia de Config (solo lectura).
    method_state : None (no se usa en esta versión sin EMA).

    Retorna
    -------
    loss     : Tensor escalar con gradiente listo para .backward().
    info     : dict con métricas de diagnóstico (sin gradiente):
                 - "lambda_min", "lambda_max", "lambda_mean"
                 - "loss_natural", "loss_robust"
               Cualquier llamante puede ignorar claves que no necesite.
    """
    loss_total, lam, loss_natural, loss_robust = d_trades_loss(
        model         = model,
        x_natural     = x,
        y             = y,
        step_size     = cfg.step_size,
        epsilon       = cfg.epsilon,
        perturb_steps = cfg.num_steps,
        alpha_base    = cfg.alpha_base,
        beta_base     = cfg.beta_base,
        gamma         = cfg.gamma,
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
# Función de pérdida D-TRADES
# ===========================================================================

def d_trades_loss(
    model,
    x_natural,
    y,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    distance='l_inf',
    alpha_base=1.0,               # Peso de H(x) en lambda
    beta_base=1.0,                # Peso de S(x) en lambda
    gamma=1.0,                    # Peso del término de error adversarial (1 - p_adv_y)
    per_sample_sensitivity=True,  # True: cálculo exacto por muestra (loop); False: aproximación
    lam_max=3.0,                  # Techo de lambda
    EPS=1e-12,                    # Estabilidad numérica para log(0) en entropía
):
    """
    Calcula la pérdida D-TRADES con lambda dinámico por muestra, sin EMA.

    Función de pérdida:
        L = L_CE(f(x), y)  +  mean( lambda(x_i) * KL(f(x_i) || f(x_i+delta)) )

    Construcción de lambda(x):
        lambda(x) = alpha_base * H_n(x)
                  + beta_base  * S_n(x)
                  + gamma      * (1 - f(x')_y)

    Donde:
        H_n(x)         -> H(x) / log(C): entropía normalizada a [0, 1].
        S_n(x)         -> log1p(S(x)): sensibilidad suavizada por logaritmo.
        (1 - f(x')_y)  -> Probabilidad de error sobre la muestra adversarial x'.
                          Siempre en [0, 1]. Bajo si el modelo clasifica bien
                          bajo ataque (no necesita más presión); alto si falla.
                          Con gamma=0 este término se desactiva.

    Por qué (1 - f(x')_y) en lugar de EMA de error:
        - Es puramente local al batch: no requiere estado entre pasos.
        - Siempre acotado en [0, 1], así que aporta a lambda de forma
          proporcional y no puede dominar artificialmente.
        - Si el modelo ya clasifica correctamente x' (p_adv_y ≈ 1),
          (1 - p_adv_y) ≈ 0 y gamma no contribuye: el lambda ya no
          empuja más. La presión viene solo de H y S en esos casos.
        - Como H_n, S_n y (1 - p_adv_y) son todos ≥ 0, lambda ≥ 0
          siempre, y la suma garantiza que nunca colapsa a cero
          mientras alguno de los tres términos sea positivo.

    Parámetros:
      model          -> Red neuronal CNN a entrenar.
      x_natural      -> Batch de imágenes limpias, forma [B, C, H, W].
      y              -> Etiquetas verdaderas, forma [B].
      step_size      -> Paso del ataque PGD.
      epsilon        -> Radio máximo de perturbación adversarial.
      perturb_steps  -> Número de pasos del ataque PGD.
      distance       -> Norma del ataque: 'l_inf' o 'l_2'.
      alpha_base     -> Peso de H_n(x) en lambda.
      beta_base      -> Peso de S_n(x) en lambda.
      gamma          -> Peso de (1 - f(x')_y) en lambda. gamma=0 lo desactiva.
      per_sample_sensitivity -> Si True, sensibilidad exacta por muestra (más lento).
      lam_max        -> Techo absoluto de lambda.
      EPS            -> Constante para evitar log(0) en entropía.

    Retorna:
      loss_total          -> Pérdida total D-TRADES (escalar, con gradiente).
      lam                 -> Lambda por muestra [B] (detached, para logging).
      loss_natural        -> Pérdida CE limpia (detached, para logging).
      loss_robust_dynamic -> Pérdida robusta ponderada (detached, para logging).
    """

    # ---------- Ataque PGD ----------
    criterion_kl_sum = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = x_natural.size(0)

    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()

    # Pre-calculamos probabilidades limpias una sola vez para el ataque
    with torch.no_grad():
        probs_nat_pgd = F.softmax(model(x_natural), dim=1)

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                loss_kl = criterion_kl_sum(
                    F.log_softmax(model(x_adv), dim=1),
                    probs_nat_pgd
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
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)

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
    # H(x) = -sum_c p_c * log(p_c)
    # Normalizada por log(C) -> rango [0, 1]
    entropy = -(probs_nat * torch.log(probs_nat.clamp_min(EPS))).sum(dim=1)
    num_classes = probs_nat.size(1)
    entropy_n = entropy / torch.log(torch.tensor(num_classes, device=entropy.device))

    # ---------- Sensibilidad adversarial S(x) [B] ----------
    # S(x) = || nabla_{x+delta} KL(f(x) || f(x+delta)) ||_2
    # Normalizada con log1p para suavizar valores extremos.
    if per_sample_sensitivity:
        # Versión exacta por muestra (loop): más lenta pero precisa.
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
        # Versión por batch (aproximación): más rápida, menos precisa por muestra.
        x_adv_req = x_adv.detach().clone().requires_grad_(True)
        # Usamos reduction='sum' para evitar dividir por B (lo que anularía este término con B grande).
        kl_total = F.kl_div(
            F.log_softmax(model(x_adv_req), dim=1),
            probs_nat.detach(),
            reduction='sum'
        )
        g = torch.autograd.grad(kl_total, x_adv_req, create_graph=False)[0]
        sensitivity = g.view(batch_size, -1).norm(p=2, dim=1)   # [B]

    sensitivity_n = torch.log1p(sensitivity)

    # ---------- Error adversarial local (1 - f(x')_y) [B] ----------
    # Probabilidad de que el modelo falle sobre la muestra adversarial x'.
    # Rango: [0, 1]. Alto -> el modelo falla bajo ataque (necesita más presión).
    #                Bajo -> el modelo clasifica bien x' (puede relajar lambda).
    # A diferencia del EMA de error, es puramente local al batch actual:
    # no acumula historia, reacciona directamente al estado del modelo ahora.
    indices = torch.arange(batch_size, device=y.device)
    adv_error = (1.0 - probs_adv[indices, y]).detach()   # [B]

    # ---------- Construcción de lambda dinámico [B] ----------
    # lambda(x) = alpha_base * H_n(x) + beta_base * S_n(x) + gamma * adv_error
    #
    # Los tres términos son >= 0, así que lambda >= 0 siempre.
    # La suma garantiza que lambda no colapsa a cero mientras algún
    # término sea positivo (el modelo no está perfectamente seguro y preciso).
    # Se detacha: lambda es peso de muestreo, no variable a optimizar.
    lam = (
        alpha_base * entropy_n
        + beta_base  * sensitivity_n
        + gamma      * adv_error
    ).detach().clamp(max=lam_max)   # [B]

    # ---------- Pérdida robusta ponderada ----------
    # L_robust = mean( lambda(x_i) * KL(f(x_i) || f(x_i+delta)) )
    loss_robust_dynamic = (lam * kl_per_example).mean()

    # Pérdida total D-TRADES
    # L = L_CE(f(x), y) + L_robust
    loss_total = loss_natural + loss_robust_dynamic

    return loss_total, lam, loss_natural.detach(), loss_robust_dynamic.detach()
