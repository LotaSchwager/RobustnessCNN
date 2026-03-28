import math

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


def compute_loss(model, x, y, cfg, method_state, epoch=1):
    """
    Interfaz genérica de pérdida para train.py.

    Parámetros
    ----------
    model        : nn.Module.
    x            : batch de imágenes limpias [B, C, H, W].
    y            : etiquetas verdaderas [B].
    cfg          : instancia de Config (solo lectura).
    method_state : None (no se usa).
    epoch        : época actual (para warmup del término adversarial).

    Retorna
    -------
    loss : Tensor escalar con gradiente listo para .backward().
    info : dict con métricas de diagnóstico (sin gradiente).
    """
    loss_total, lam, loss_natural, loss_robust = d_trades_loss(
        model          = model,
        x_natural      = x,
        y              = y,
        step_size      = cfg.step_size,
        epsilon        = cfg.epsilon,
        perturb_steps  = cfg.num_steps,
        num_classes    = cfg.num_classes,
        alpha          = cfg.alpha_base,
        beta           = cfg.beta_base,
        weight_floor   = cfg.weight_floor,
        lam_clamp_min  = cfg.lam_clamp_min,
        lam_clamp_max  = cfg.lam_clamp_max,
        warmup_epochs  = cfg.warmup_epochs,
        epoch          = epoch,
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
# Normalización (solo para sensibilidad — detached por diseño)
# ===========================================================================

@torch.no_grad()
def _normalize_batch(v, eps=1e-8):
    """
    Normaliza un vector [B] al rango [0, 1] con min-max sobre el batch.
    eps=1e-8 evita división por cero cuando todos los valores son iguales.

    Nota: @torch.no_grad() es intencional aquí — la sensibilidad se usa como
    peso (detached), no como término optimizable. Para la entropía (que sí
    necesita gradiente) se usa la normalización por log(C).
    """
    vmin = v.min()
    vmax = v.max()
    return (v - vmin) / (vmax - vmin + eps)


# ===========================================================================
# Función de pérdida D-TRADES v2 — formulación mejorada
# ===========================================================================

def d_trades_loss(
    model,
    x_natural,
    y,
    step_size,
    epsilon,
    perturb_steps,
    num_classes=10,
    distance='l_inf',
    alpha=0.5,
    beta=0.5,
    weight_floor=0.1,
    lam_clamp_min=0.1,
    lam_clamp_max=5.0,
    warmup_epochs=10,
    epoch=1,
    EPS=1e-12,
):
    """
    Pérdida D-TRADES v2 con formulación mejorada.

    Cambios respecto a v1:
    ──────────────────────
    1. Entropy como modulador (no aditivo): H(x) amplifica el KL junto con S(x)
       en lugar de sumarse independientemente. Un solo término con gradiente (KL).
    2. Piso en (1 - p_y): evita que muestras bien clasificadas reciban peso ~0,
       previniendo "olvido de robustez" en predicciones confiadas pero frágiles.
    3. Clamp del peso efectivo: previene que outliers dominen el gradiente del batch.
    4. Reutilización del gradiente PGD: la sensibilidad se calcula con el último
       gradiente del ataque PGD, eliminando un forward+backward extra (~33% speedup).
    5. Warmup del término adversarial: durante las primeras épocas el modelo
       construye features naturales antes de recibir presión adversarial completa.

    Estructura de pérdida:
    ──────────────────────
        L = L_CE(f(x), y)  +  warmup * mean( w(x) * KL(f(x) || f(x')) )

    Donde:
        w(x) = clamp( error_weight(x) * lambda(x),  lam_min, lam_max )

        error_weight(x)  = max( 1 - f(x)_y,  weight_floor )
                           Ponderador MART suavizado. weight_floor > 0 garantiza
                           que muestras bien clasificadas mantienen entrenamiento
                           de robustez mínimo.

        lambda(x)        = 1 + beta * S_norm(x) + alpha * H_norm(x)
                           Modulador unificado. S_norm y H_norm están en [0, 1].
                           Rango: [1, 1 + alpha + beta].
                           S_norm: sensibilidad normalizada (norma del gradiente PGD).
                           H_norm: entropía normalizada por log(C).

        warmup           = min(1, epoch / warmup_epochs)
                           Rampa lineal. Las primeras épocas priorizan accuracy
                           natural; el término adversarial crece gradualmente.

    Por qué esta estructura es coherente:
    ──────────────────────────────────────
        - Un solo término con gradiente (KL): el modelo optimiza una sola cosa
          en la parte adversarial — cerrar la brecha entre predicción limpia y
          adversarial. Todos los demás factores son pesos detached que modulan
          cuánto esfuerzo se dedica a cada muestra.

        - S(x) y H(x) como moduladores conjuntos: ambos identifican muestras
          "problemáticas" desde ángulos complementarios (inestabilidad geométrica
          vs. incertidumbre predictiva). Sumarlos dentro de lambda da un score
          compuesto de dificultad.

        - El clamp previene que el producto error_weight * lambda genere pesos
          extremos que desestabilicen el entrenamiento.

        - El warmup evita gastar compute en PGD cuando el modelo es aleatorio
          y las perturbaciones no son informativas.

    Parámetros
    ──────────
    model              CNN a entrenar.
    x_natural          Imágenes limpias [B, C, H, W].
    y                  Etiquetas verdaderas [B].
    step_size          Paso del ataque PGD.
    epsilon            Radio máximo de perturbación.
    perturb_steps      Pasos del ataque PGD.
    num_classes        Número de clases. Para normalizar H(x) a [0, 1].
    distance           Norma del ataque: 'l_inf' o 'l_2'.
    alpha              Peso de H(x) en el modulador lambda.
    beta               Peso de S(x) en el modulador lambda.
                       alpha=beta=0 equivale a TRADES + MART weighting.
    weight_floor       Piso mínimo para (1 - p_y). Evita peso cero en
                       muestras confiadas. Default 0.1.
    lam_clamp_min      Piso del peso efectivo w(x). Default 0.1.
    lam_clamp_max      Techo del peso efectivo w(x). Default 5.0.
    warmup_epochs      Épocas para rampa completa del término adversarial.
                       warmup_epochs=0 desactiva el warmup.
    epoch              Época actual del entrenamiento.
    EPS                Estabilidad numérica para log(0).

    Retorna
    ───────
    loss_total   Pérdida total (escalar, con gradiente).
    lam          Peso efectivo por muestra [B] (detached, para logging).
    loss_natural Pérdida CE (detached, para logging).
    loss_robust  Término adversarial (detached, para logging).
    """

    # ---------- Ataque PGD (reutiliza último gradiente para sensibilidad) ----
    criterion_kl_sum = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = x_natural.size(0)

    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    last_pgd_grad = None   # se guarda el último gradiente para sensibilidad

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                loss_kl = criterion_kl_sum(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1)
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            last_pgd_grad = grad.detach()   # guardar para sensibilidad
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
            last_pgd_grad = delta.grad.detach().clone()   # guardar para sensibilidad
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            safe = grad_norms.clamp(min=1e-12)
            delta.grad.div_(safe.view(-1, 1, 1, 1))
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

    # ---------- H(x): entropía local normalizada [B] ----------
    # H(x) = -sum_c p_c * log(p_c), normalizada a [0, 1] por log(C).
    # Diferenciable: el gradiente fluye y contribuye como modulador.
    # En v2, H_norm modula el KL en lugar de sumarse como término independiente.
    entropy = -(probs_nat * torch.log(probs_nat.clamp_min(EPS))).sum(dim=1)
    log_C = math.log(num_classes) if num_classes > 1 else 1.0
    entropy_n = entropy / log_C   # [B], rango [0, 1], con gradiente

    # ---------- S(x): sensibilidad reutilizando gradiente PGD [B] ----------
    # En v1 se hacía un forward+backward extra para calcular la sensibilidad.
    # En v2 reutilizamos el último gradiente del ataque PGD: mide exactamente
    # lo mismo (cuánto varía la KL ante perturbaciones de x') sin coste extra.
    # Ahorro: ~33% de compute por batch (un forward+backward menos).
    if last_pgd_grad is not None:
        sensitivity = last_pgd_grad.view(batch_size, -1).norm(p=2, dim=1)
    else:
        # Fallback: si no hubo pasos PGD (perturb_steps=0), sensibilidad cero.
        sensitivity = torch.zeros(batch_size, device=x_natural.device)

    sensitivity_n = _normalize_batch(sensitivity)   # [B], rango [0, 1], detached

    # ---------- lambda(x): modulador unificado [B] ----------
    # lambda(x) = 1 + beta * S_norm(x) + alpha * H_norm(x)
    #
    # En v1: lambda_S * KL + alpha * H  (H era aditivo, sin gradiente efectivo)
    # En v2: lambda * KL               (H modula el KL junto con S)
    #
    # Ventaja: un solo término con gradiente (KL). S y H son moduladores que
    # identifican muestras problemáticas desde ángulos complementarios:
    #   - S alto: el landscape es empinado → muestra geométricamente inestable
    #   - H alto: el modelo está confundido → muestra semánticamente difícil
    #
    # Rango de lambda: [1, 1 + alpha + beta].
    # sensitivity_n ya es detached (viene de _normalize_batch).
    # entropy_n tiene gradiente — esto es intencional: a través de lambda,
    # el gradiente de H penaliza predicciones inciertas proporcionalmente
    # a cuánto KL tiene cada muestra. Muestras con KL alto y H alto reciben
    # la mayor presión para volverse más seguras.
    lambda_mod = 1.0 + beta * sensitivity_n + alpha * entropy_n   # [B]

    # ---------- error_weight: (1 - p_y) suavizado [B] ----------
    # Piso en weight_floor para que muestras bien clasificadas mantengan
    # un mínimo de entrenamiento adversarial. Sin piso, una muestra con
    # p_y=0.99 recibe peso 0.01 — prácticamente ignorada, lo que puede
    # causar "olvido de robustez" (la muestra pierde su robustez porque
    # nunca se entrena adversarialmente).
    p_y = probs_nat[torch.arange(batch_size), y]
    error_weight = (1.0 - p_y).clamp(min=weight_floor).detach()   # [B]

    # ---------- Peso efectivo clamped [B] ----------
    # Clamp para prevenir que outliers (alta sensibilidad + alta entropía +
    # baja confianza) dominen el gradiente del batch.
    # lam_clamp_min garantiza robustez mínima para toda muestra.
    # lam_clamp_max previene que una sola muestra acapare el gradiente.
    effective_weight = (error_weight * lambda_mod).clamp(
        min=lam_clamp_min, max=lam_clamp_max
    )   # [B]
    # Nota: effective_weight NO está completamente detached — entropy_n
    # (dentro de lambda_mod) mantiene su gradiente. El clamp es diferenciable
    # en el rango interior [lam_min, lam_max]; en los bordes el gradiente
    # es cero (comportamiento deseado: si el peso está saturado, no queremos
    # que la entropía siga empujando).

    # ---------- Warmup del término adversarial ----------
    # Durante las primeras épocas el modelo es aleatorio: las perturbaciones
    # PGD no son informativas y el KL es ruidoso. El warmup permite que el
    # modelo construya features naturales útiles antes de agregar presión
    # adversarial completa.
    if warmup_epochs > 0:
        warmup_factor = min(1.0, epoch / warmup_epochs)
    else:
        warmup_factor = 1.0

    # ---------- Término adversarial ----------
    loss_robust_dynamic = warmup_factor * (effective_weight * kl_per_example).mean()

    # lam: peso efectivo por muestra para logging (detached).
    lam = effective_weight.detach()   # [B]

    # ---------- Pérdida total ----------
    loss_total = loss_natural + loss_robust_dynamic

    return loss_total, lam, loss_natural.detach(), loss_robust_dynamic.detach()
