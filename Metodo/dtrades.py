import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# ===========================================================================
# Interfaz pública genérica (registrada en Metodo/__init__.py)
# ===========================================================================
# Todo método debe exponer estas cuatro funciones con esta firma exacta.
# Así train.py y main.py son completamente agnósticos al método activo.

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
    model        : nn.Module.
    x            : batch de imágenes limpias [B, C, H, W].
    y            : etiquetas verdaderas [B].
    cfg          : instancia de Config (solo lectura).
    method_state : None (no se usa en esta versión).

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
        alpha         = cfg.alpha_base,       # peso de H(x) en el modulador
        beta          = cfg.beta_base,        # peso de S(x) en el modulador
        gamma         = cfg.gamma,            # peso de interacción H*S
        weight_floor  = cfg.weight_floor,     # piso de (1 - f(x')_y)
        lam_max       = cfg.lam_max,          # techo absoluto de lambda
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
# Normalización por media — más estable que min-max
# ===========================================================================

@torch.no_grad()
def _normalize_by_mean(v, eps=1e-8):
    """
    Normaliza un vector [B] dividiéndolo por su media.

    Por qué media en lugar de min-max:
      - Min-max colapsa a ~0 cuando hay poca varianza en el batch (todas las
        muestras tienen sensibilidad similar). La media siempre refleja la
        magnitud relativa: si S(x_i) = 1.5 * media, esa muestra es 50% más
        sensible que el promedio, independientemente de los valores extremos.
      - Es más robusta ante outliers: un valor extremo desplaza la media
        modestamente, pero en min-max un solo outlier comprime todo el rango.

    El resultado no está acotado en [0,1] — una muestra puede dar >1 si
    supera la media. El techo lo controla lam_max en la función principal.

    eps=1e-8 evita división por cero si todos los valores son iguales a 0
    (ej. primeras iteraciones donde la sensibilidad es mínima).
    """
    return v / (v.mean() + eps)


# ===========================================================================
# Función de pérdida D-TRADES — formulación integrada
# ===========================================================================

def d_trades_loss(
    model,
    x_natural,
    y,
    step_size,
    epsilon,
    perturb_steps,
    distance      = 'l_inf',
    alpha         = 1.0,
    beta          = 1.0,
    gamma         = 1.0,
    weight_floor  = 0.1,
    lam_max       = 3.5,
    lam_min       = 0.1,
    EPS           = 1e-12,
):
    """
    Pérdida D-TRADES con lambda dinámico por muestra.

    ── Estructura de pérdida ─────────────────────────────────────────────────

        L = L_CE(f(x), y)  +  mean( lambda(x_i) * KL(f(x_i) || f(x_i')) )

    ── Construcción de lambda(x) ─────────────────────────────────────────────

        lambda(x) = clamp(
            max(1 - f(x')_y, weight_floor)  *  (1 + beta*S_n + alpha*H_n + gamma*H_n*S_n),
            0, lam_max
        )

    Donde cada factor tiene un rol semántico claro:

      max(1 - f(x')_y, weight_floor)
          Ponderador de dificultad adversarial inspirado en MART
          (Wang et al., 2020). Usa la predicción sobre la muestra
          ADVERSARIAL x', no la limpia x, por dos razones:
            1. Captura la vulnerabilidad real bajo ataque: si el modelo
               predice bien x' (alta p_y), la muestra ya está protegida
               y necesita menos presión adversarial.
            2. Evita el incentivo perverso de la versión anterior: con
               f(x)_y el modelo podía subir p_y en limpio para reducir
               lambda sin mejorar la robustez. Con f(x')_y el modelo
               solo reduce lambda siendo genuinamente robusto bajo ataque.
          weight_floor > 0 garantiza que muestras bien clasificadas bajo
          ataque aún reciben entrenamiento adversarial mínimo, evitando
          "olvido de robustez" en muestras confiadas pero frágiles.
          Rango: [weight_floor, 1].

      (1 + beta*S_n + alpha*H_n + gamma*H_n*S_n)
          Modulador de riesgo compuesto. Amplifica el KL para muestras
          que son simultáneamente inciertas y sensibles.
            - S_n = log1p(S(x)): sensibilidad normalizada usando log1p.
              Mide la inestabilidad geométrica local de manera más estable,
              suavizando valores extremos.
            - H_n = H(x) / log(C): entropía normalizada a su valor máximo
              teórico. Mide la incertidumbre predictiva en el rango [0, 1].
            - H_n * S_n: término de interacción (gamma=0 lo desactiva).
              Captura el caso extremo: muestra incierta Y sensible, que
              es el escenario más peligroso bajo ataque adversarial.
          Rango teórico: >= 1 (piso garantizado por el 1).

      clamp(..., 0, lam_max)
          Techo absoluto que previene que outliers extremos dominen el
          gradiente del batch. Si una muestra tiene S y H muy altos, sin
          techo lambda crecería sin límite. lam_max controla el rango
          final de lambda independientemente de la escala de S y H.

    ── Por qué esta estructura es coherente ──────────────────────────────────

      - Un único término con gradiente: solo KL está dentro del grafo de
        backprop. Lambda es un peso detached — el modelo optimiza cerrar la
        brecha entre predicción limpia y adversarial, no minimizar entropía
        directamente (lo que colapsa el modelo hacia predicciones artificialmente
        seguras).

      - Normalización natural: H(x) acotado dividiendo por log(C) y
        S(x) suavizado con log1p para estabilizar los rangos y mantener coherencia.

      - f(x')_y en lugar de f(x)_y: elimina el incentivo perverso que causó
        el colapso completo en versiones anteriores.

    ── Parámetros ────────────────────────────────────────────────────────────

    model         CNN a entrenar.
    x_natural     Imágenes limpias [B, C, H, W].
    y             Etiquetas verdaderas [B].
    step_size     Paso del ataque PGD.
    epsilon       Radio máximo de perturbación.
    perturb_steps Pasos del ataque PGD.
    distance      Norma del ataque: 'l_inf' o 'l_2'.
    alpha         Peso de H(x) en el modulador. alpha=0: H no contribuye.
    beta          Peso de S(x) en el modulador. beta=0: S no contribuye.
    gamma         Peso del término de interacción H*S. gamma=0: desactivado.
                  Útil para ablaciones: probar con gamma=0 primero y subir
                  si los experimentos muestran beneficio.
    weight_floor  Piso mínimo del ponderador de dificultad. Default 0.1.
                  Previene que muestras confiadas bajo ataque reciban λ≈0.
    lam_max       Techo de lambda tras el clamp. Default 2.0.
                  Valor de referencia: TRADES usa lambda fijo en [1, 6],
                  equivalente a 1/beta en [0.17, 1]. lam_max=2 es
                  conservador y comparable con el rango histórico observado.
    EPS           Estabilidad numérica para log(0) en entropía.

    ── Retorna ───────────────────────────────────────────────────────────────

    loss_total   Pérdida total (escalar, con gradiente).
    lam          Lambda por muestra [B] (detached, para logging).
    loss_natural Pérdida CE sobre muestras limpias (detached, para logging).
    loss_robust  Término adversarial ponderado (detached, para logging).
    """

    # ── Ataque PGD ────────────────────────────────────────────────────────────
    # Generamos la muestra adversarial x' que maximiza la KL dentro de epsilon.
    # El modelo se pone en eval() durante el ataque para que BatchNorm use
    # estadísticas fijas — comportamiento estándar en TRADES.
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

    # ── Forward limpio y adversarial ─────────────────────────────────────────
    logits_nat    = model(x_natural)
    logits_adv    = model(x_adv.detach())

    probs_nat     = F.softmax(logits_nat, dim=1).clamp(min=1e-8, max=1.0)
    probs_adv     = F.softmax(logits_adv, dim=1).clamp(min=1e-8, max=1.0)
    log_probs_adv = torch.log(probs_adv).clamp(min=-50, max=0)

    # Pérdida de clasificación sobre muestras limpias
    loss_natural = F.cross_entropy(logits_nat, y, reduction='mean')

    # KL por muestra [B]: KL(f(x) || f(x'))
    # Mide cuánto cambia la predicción del modelo al aplicar la perturbación.
    kl_per_example = F.kl_div(
        log_probs_adv, probs_nat, reduction='none'
    ).sum(dim=1)

    # ── Entropía local H(x) [B] ───────────────────────────────────────────────
    # H(x) = -sum_c p_c * log(p_c)
    # Captura la incertidumbre del modelo sobre la muestra limpia.
    # Alta cuando la distribución es plana (modelo confundido).
    # Baja cuando está concentrada en una clase (modelo seguro).
    entropy = -(probs_nat * torch.log(probs_nat.clamp_min(EPS))).sum(dim=1)  # [B]

    # ── Sensibilidad adversarial S(x) [B] ────────────────────────────────────
    # S(x) = || nabla_{x'} KL(f(x) || f(x')) ||_2
    # Mide la inestabilidad geométrica: cuánto varía la KL ante perturbaciones
    # de x'. Un valor alto indica que la muestra está en una región del espacio
    # de entrada donde la robustez es especialmente frágil.
    # Usamos aproximación por batch (un único backward sobre KL promedio):
    # es suficientemente precisa para el modulador y evita el loop de
    # batch_size backwards individuales que era el cuello de botella anterior.
    x_adv_req = x_adv.detach().clone().requires_grad_(True)
    kl_total  = F.kl_div(
        F.log_softmax(model(x_adv_req), dim=1),
        probs_nat.detach(),
        reduction='batchmean'
    )
    g = torch.autograd.grad(kl_total, x_adv_req, create_graph=False)[0]
    sensitivity = g.view(batch_size, -1).norm(p=2, dim=1)  # [B]

    # ── Normalización de entropía y sensibilidad ──────────────────────────────
    # Entropía: se normaliza a su valor natural dividiendo por log(C).
    # Sensibilidad: se normaliza usando log1p para mayor estabilidad.
    entropy_n     = entropy.detach() / math.log(probs_nat.size(1))
    sensitivity_n = torch.log1p(sensitivity.detach())

    # ── Ponderador de dificultad adversarial (inspirado en MART) ─────────────
    # (1 - f(x')_y): probabilidad de error sobre la muestra ADVERSARIAL.
    # Si el modelo ya clasifica bien x' (p_y alto), la muestra recibe poco peso.
    # Si el modelo falla bajo ataque (p_y bajo), la muestra recibe mucho peso.
    # El weight_floor evita que muestras confiadas bajo ataque reciban peso ~0.
    p_y_adv      = probs_adv[torch.arange(batch_size), y]              # [B]
    error_weight = (1.0 - p_y_adv).clamp(min=weight_floor).detach()   # [B]

    # ── Modulador de riesgo compuesto ─────────────────────────────────────────
    # (1 + beta*S_n + alpha*H_n + gamma*H_n*S_n)
    # Piso en 1: el KL nunca desaparece aunque S y H sean bajos.
    # El término gamma*H_n*S_n amplifica extra cuando ambos son altos
    # simultáneamente (muestra incierta Y geométricamente inestable).
    # modulator = 1.0 + beta * sensitivity_n + alpha * entropy_n + gamma * entropy_n * sensitivity_n  # [B]
    modulator = beta * sensitivity_n + alpha * entropy_n + gamma * entropy_n * sensitivity_n  # [B]

    # ── Lambda dinámico final [B] ─────────────────────────────────────────────
    # lambda(x) = clamp( error_weight * modulator, 0, lam_max )
    # Detached: lambda es un peso de muestreo, no un término a optimizar.
    # El modelo optimiza cerrar la brecha clean-adversarial (KL), no reducir H o S.
    lam = (error_weight * modulator).clamp(max=lam_max, min=lam_min).detach()  # [B]

    lam = lam /(lam.mean() + 1e-8)

    # ── Pérdida adversarial ponderada ─────────────────────────────────────────
    # L_robust = mean( lambda(x_i) * KL(f(x_i) || f(x_i')) )
    loss_robust_dynamic = (lam * kl_per_example).mean()

    # L_total = L_CE(f(x), y) + L_robust
    loss_total = loss_natural + loss_robust_dynamic

    return loss_total, lam, loss_natural.detach(), loss_robust_dynamic.detach()
