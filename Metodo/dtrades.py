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
    D-TRADES (sin EMA) no requiere estado persistente entre batches.
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
    model        : nn.Module.
    x            : batch de imágenes limpias [B, C, H, W].
    y            : etiquetas verdaderas [B].
    cfg          : instancia de Config (solo lectura).
    method_state : None (no se usa en esta versión sin EMA).

    Retorna
    -------
    loss : Tensor escalar con gradiente listo para .backward().
    info : dict con métricas de diagnóstico (sin gradiente).
    """
    loss_total, lam, loss_natural, loss_robust, alpha_cls, beta_cls = d_trades_loss(
        model         = model,
        x_natural     = x,
        y             = y,
        step_size     = cfg.step_size,
        epsilon       = cfg.epsilon,
        perturb_steps = cfg.num_steps,
        num_classes   = cfg.num_classes,
        alpha_base    = cfg.alpha_base,
        beta_base     = cfg.beta_base,
        gamma         = cfg.gamma,
    )
    info = {
        "lambda_min":      lam.min().item(),
        "lambda_max":      lam.max().item(),
        "lambda_mean":     lam.mean().item(),
        "loss_natural":    loss_natural.item(),
        "loss_robust":     loss_robust.item(),
        "alpha_per_class": alpha_cls,
        "beta_per_class":  beta_cls,
    }
    return loss_total, info


# ===========================================================================
# Funciones de normalización
# ===========================================================================

@torch.no_grad()
def _normalize_batch(v, eps=1e-8):
    """
    Normaliza un vector [B] al rango [0, 1] con min-max sobre el batch.
    eps=1e-8 evita division por cero cuando todos los valores son iguales.
    """
    vmin = v.min()
    vmax = v.max()
    return (v - vmin) / (vmax - vmin + eps)


@torch.no_grad()
def _normalize_class_stats(v, eps=1e-8):
    """
    Normaliza un vector [num_classes] al rango [0, 1] con min-max entre clases.
    Se usa para normalizar H_c y S_c antes de calcular alpha_c y beta_c.
    eps=1e-8 evita division por cero cuando todas las clases tienen el mismo valor.
    """
    vmin = v.min()
    vmax = v.max()
    return (v - vmin) / (vmax - vmin + eps)


# ===========================================================================
# Estadísticas por clase calculadas sobre el batch actual (sin EMA)
# ===========================================================================

@torch.no_grad()
def _batch_class_stats(y, entropy, sensitivity, logits_nat, num_classes):
    """
    Calcula H_c, S_c y err_c directamente sobre las muestras del batch actual.
    No hay memoria de batches anteriores — cada batch es completamente independiente.

    Para cada clase c presente en el batch:
      H_c   = media de entropía de las muestras de clase c en este batch
      S_c   = media de sensibilidad de las muestras de clase c en este batch
      err_c = tasa de error de las muestras de clase c en este batch

    Para clases sin muestras en el batch, los valores quedan en 0.0.
    Con batch=256 y 10 clases hay ~25 muestras por clase en promedio,
    suficiente para una estimación razonable por batch.

    Parámetros:
      y           -> Etiquetas verdaderas [B].
      entropy     -> Entropia local por muestra [B].
      sensitivity -> Sensibilidad local por muestra [B].
      logits_nat  -> Logits del modelo sobre muestras limpias [B, C].
      num_classes -> Numero total de clases del dataset.

    Retorna:
      H_c   -> Tensor [num_classes], entropia media por clase en el batch.
      S_c   -> Tensor [num_classes], sensibilidad media por clase en el batch.
      err_c -> Tensor [num_classes], tasa de error por clase en el batch.
    """
    device = y.device
    H_c   = torch.zeros(num_classes, device=device)
    S_c   = torch.zeros(num_classes, device=device)
    err_c = torch.zeros(num_classes, device=device)

    preds = logits_nat.argmax(dim=1)   # [B]

    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() == 0:
            continue
        H_c[c]   = entropy[mask].mean()
        S_c[c]   = sensitivity[mask].mean()
        err_c[c] = 1.0 - (preds[mask] == c).float().mean()

    return H_c, S_c, err_c


# ===========================================================================
# Funcion de perdida D-TRADES (sin EMA)
# ===========================================================================

def d_trades_loss(
    model,
    x_natural,
    y,
    step_size,
    epsilon,
    perturb_steps,
    num_classes,
    distance='l_inf',
    alpha_base=0.5,
    beta_base=0.5,
    gamma=0.25,
    normalize_terms=True,
    per_sample_sensitivity=False,
    EPS=1e-12,
):
    """
    Calcula la perdida D-TRADES con lambda dinamico por batch (sin EMA).

    Funcion de perdida:
        L_D-TRADES(f, x, y) = L_CE(f(x), y) + lambda(x) * KL(f(x) || f(x+delta))

    Lambda dinamico por muestra:
        lambda(x_i) = alpha_c * H(x_i) + beta_c * S(x_i) + gamma * err_c

    Con (Opcion B, piso garantizado):
        alpha_c = alpha_base * (1 + H_tilde_c)   rango [alpha_base, 2*alpha_base]
        beta_c  = beta_base  * (1 + S_tilde_c)   rango [beta_base,  2*beta_base]

    H_tilde_c y S_tilde_c son normalizaciones min-max de H_c y S_c
    calculadas sobre las clases presentes en el batch actual (sin EMA).

    Diferencia respecto a la version con EMA:
      - alpha_c y beta_c reflejan el estado del batch actual, no el historico.
      - Mas reactivo pero mas ruidoso batch a batch.
      - No requiere estado persistente (method_state=None).

    Parámetros:
      model                  -> CNN a entrenar.
      x_natural              -> Imagenes limpias [B, C, H, W].
      y                      -> Etiquetas verdaderas [B].
      step_size              -> Paso del ataque PGD.
      epsilon                -> Radio maximo de perturbacion.
      perturb_steps          -> Pasos del ataque PGD.
      num_classes            -> Numero de clases del dataset.
      distance               -> Norma del ataque: 'l_inf' o 'l_2'.
      alpha_base             -> Peso base para la entropia.
      beta_base              -> Peso base para la sensibilidad.
      gamma                  -> Peso del error de clase en lambda.
      normalize_terms        -> Si True, normaliza H(x) y S(x) locales a [0,1].
      per_sample_sensitivity -> Si True, sensibilidad exacta por muestra (loop).
                                Si False, aproximacion por batch (recomendado).
      EPS                    -> Estabilidad numerica para log(0) en entropia.

    Retorna:
      loss_total   -> Perdida total (escalar, con gradiente).
      lam          -> Lambda por muestra [B] (detached).
      loss_natural -> Perdida de clasificacion (detached, para logging).
      loss_robust  -> Perdida robusta ponderada (detached, para logging).
      alpha_cls    -> list[float] de alpha_c por clase (para metricas).
      beta_cls     -> list[float] de beta_c por clase (para metricas).
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

    # ---------- Forward limpio ----------
    logits_nat    = model(x_natural)
    logits_adv    = model(x_adv.detach())

    probs_nat     = F.softmax(logits_nat, dim=1).clamp(min=1e-8, max=1.0)
    log_probs_adv = F.log_softmax(logits_adv, dim=1).clamp(min=-50, max=0)

    loss_natural = F.cross_entropy(logits_nat, y, reduction='mean')

    # KL por muestra [B]
    kl_per_example = F.kl_div(
        log_probs_adv, probs_nat, reduction='none'
    ).sum(dim=1)

    # ---------- Entropia local H(x) [B] ----------
    # H(x) = -sum_c p_c * log(p_c)
    # Captura la incertidumbre del modelo sobre cada muestra.
    entropy = -(probs_nat * torch.log(probs_nat.clamp_min(EPS))).sum(dim=1)

    # ---------- Sensibilidad local S(x) [B] ----------
    # S(x) = || nabla_{x+delta} KL(f(x) || f(x+delta)) ||_2
    if per_sample_sensitivity:
        # Exacta por muestra: loop de backwards individuales.
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
        sensitivity = torch.cat(sens_list, dim=0)
        model.train()
    else:
        # Aproximacion por batch: un unico backward sobre la KL promedio.
        x_adv_req = x_adv.detach().clone().requires_grad_(True)
        kl_total  = F.kl_div(
            F.log_softmax(model(x_adv_req), dim=1),
            probs_nat.detach(),
            reduction='batchmean'
        )
        g = torch.autograd.grad(kl_total, x_adv_req, create_graph=False)[0]
        sensitivity = g.view(batch_size, -1).norm(p=2, dim=1)

    # ---------- Normalizacion local [B] -> [0, 1] ----------
    if normalize_terms:
        entropy_n     = _normalize_batch(entropy)
        sensitivity_n = _normalize_batch(sensitivity)
    else:
        entropy_n     = entropy
        sensitivity_n = sensitivity

    # ---------- Estadisticas por clase sobre el batch actual ----------
    # Sin EMA: solo usa muestras del batch presente.
    H_c, S_c, err_c = _batch_class_stats(
        y, entropy.detach(), sensitivity.detach(), logits_nat.detach(), num_classes
    )

    # ---------- alpha_c y beta_c por clase (Opcion B) ----------
    H_tilde = _normalize_class_stats(H_c)
    S_tilde = _normalize_class_stats(S_c)

    alpha_c = alpha_base * (1.0 + H_tilde)   # [num_classes]
    beta_c  = beta_base  * (1.0 + S_tilde)   # [num_classes]

    alpha_per_sample = alpha_c[y]   # [B]
    beta_per_sample  = beta_c[y]    # [B]
    err_per_sample   = err_c[y]     # [B]

    # ---------- Lambda dinamico [B] ----------
    # lambda(x_i) = alpha_c * H(x_i) + beta_c * S(x_i) + gamma * err_c
    lam = (
        alpha_per_sample * entropy_n
        + beta_per_sample  * sensitivity_n
        + gamma            * err_per_sample
    ).detach()

    # ---------- Perdida robusta ponderada ----------
    loss_robust_dynamic = (lam * kl_per_example).mean()
    loss_total = loss_natural + loss_robust_dynamic

    return (
        loss_total,
        lam,
        loss_natural.detach(),
        loss_robust_dynamic.detach(),
        alpha_c.tolist(),
        beta_c.tolist(),
    )
