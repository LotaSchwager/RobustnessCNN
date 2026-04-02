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

def make_state(cfg, device) -> "ClassStats":
    """
    Crea el estado mutable del método (ClassStats para D-TRADES).
    Llamado una única vez en main.py, antes del loop de entrenamiento.
    El estado persiste entre épocas pasado por referencia.
    """
    num_classes = cfg.num_classes   # resuelto desde DataConfig en main.py
    rho         = cfg.rho
    return ClassStats(num_classes=num_classes, rho=rho, device=device)


def save_state(state: "ClassStats", path: str) -> None:
    """Serializa el estado EMA a disco junto al checkpoint del modelo."""
    stats_path = path + ".stats"
    print(f"  -> Guardando estadisticas de clase: {stats_path}")
    torch.save(
        {"H_c": state.H_c.cpu(), "S_c": state.S_c.cpu(), "err_c": state.err_c.cpu()},
        stats_path,
    )


def load_state(state: "ClassStats", path: str, device) -> None:
    """Restaura el estado EMA desde disco (in-place)."""
    import os
    stats_path = path + ".stats"
    if os.path.exists(stats_path):
        saved = torch.load(stats_path, map_location=device)
        state.H_c   = saved["H_c"].to(device)
        state.S_c   = saved["S_c"].to(device)
        state.err_c = saved["err_c"].to(device)
        print(f"[RESUME] ClassStats.EMA restaurado desde: {stats_path}")
    else:
        print(f"[RESUME] No se encontró {stats_path} — ClassStats inicia desde cero.")


def compute_loss(model, x, y, cfg, method_state: "ClassStats"):
    """
    Interfaz genérica de pérdida para train.py.

    Parámetros
    ----------
    model        : nn.Module en modo train/eval según requiera el método.
    x            : batch de imágenes limpias [B, C, H, W].
    y            : etiquetas verdaderas [B].
    cfg          : instancia de Config (solo lectura).
    method_state : ClassStats (estado mutable de D-TRADES).

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
        optimizer     = None,        # no se usa internamente en d_trades_loss
        class_stats   = method_state,
        step_size     = cfg.step_size,
        epsilon       = cfg.epsilon,
        perturb_steps = cfg.num_steps,
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
        # α_c y β_c por clase: valores EMA actualizados de ClassStats.
        # Se convierten a lista de floats Python para que Metrics los serialice.
        "alpha_per_class": method_state.get_alpha_beta_class(cfg.alpha_base, cfg.beta_base)[0],
        "beta_per_class":  method_state.get_alpha_beta_class(cfg.alpha_base, cfg.beta_base)[1],
    }
    return loss_total, info



# ===========================================================================
# Funciones de normalización
# ===========================================================================

"""
Normaliza un vector al rango [0, 1] usando min-max sobre el batch completo.
Parámetros:
  v   -> Tensor 1D de forma [B].
  eps -> Constante de estabilidad numérica para evitar división por cero
         cuando todos los valores son iguales (denominador = 0).
         Valor estándar: 1e-8 (mismo que usa Adam en PyTorch).
"""
@torch.no_grad()
def _normalize_batch(v, eps=1e-8):
    vmin = v.min()
    vmax = v.max()
    return (v - vmin) / (vmax - vmin + eps)


"""
Normaliza un vector al rango [0, 1] usando min-max sobre las clases.
Se usa para normalizar las estadísticas globales por clase H_c y S_c
antes de calcular alpha_c y beta_c.
Parámetros:
  v   -> Tensor 1D de forma [num_classes], uno por clase.
  eps -> Constante de estabilidad numérica (1e-8 por defecto).
"""
@torch.no_grad()
def _normalize_class_stats(v, eps=1e-8):
    vmin = v.min()
    vmax = v.max()
    return (v - vmin) / (vmax - vmin + eps)


# ===========================================================================
# Clase de estadísticas por clase (EMA)
# ===========================================================================

class ClassStats:
    """
    Mantiene estadísticas globales por clase actualizadas mediante EMA
    (Exponential Moving Average) a lo largo del entrenamiento.

    Las estadísticas acumuladas son:
      - H_c  : entropía media de las muestras de clase c  -> incertidumbre global de la clase
      - S_c  : sensibilidad adversarial media de clase c  -> vulnerabilidad global de la clase
      - err_c: tasa de error de clase c                   -> dificultad de clasificación global

    La fórmula EMA es:
        v_c^(t) = (1 - rho) * v_c^(t-1) + rho * v_c_batch^(t)

    Donde rho es la tasa de actualización:
      - rho alto (ej. 0.9): mucho peso a la observación reciente, olvida rápido el pasado.
      - rho bajo (ej. 0.1): suaviza más, recuerda más historia.
      Valor típico: 0.1 para estadísticas de clase en entrenamiento.

    Inicialización:
      - H_c  = 0.0 : sin incertidumbre observada aún.
      - S_c  = 0.0 : sin sensibilidad observada aún.
      - err_c = 0.0: sin error observado aún (alternativa conservadora: 1.0).

    Si en un batch no hay muestras de clase c, esa clase NO se actualiza
    ese paso, manteniendo su valor anterior intacto.

    Referencia del mecanismo EMA:
      - Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization"
        usa EMA para los momentos de primer y segundo orden.
      - Ioffe & Szegedy (2015) "Batch Normalization" usa EMA para
        mantener media y varianza globales durante el entrenamiento.
    """

    def __init__(self, num_classes: int, rho: float = 0.1, device=None):
        """
        Parámetros:
          num_classes -> Número total de clases del dataset (ej. 10 para CIFAR-10).
          rho         -> Tasa de actualización EMA. Default: 0.1.
          device      -> Dispositivo donde viven los tensores (CPU o CUDA).
        """
        self.num_classes = num_classes
        self.rho = rho
        self.device = device

        # Estadísticas acumuladas por clase, forma [num_classes]
        # Inicializadas en 0 — el primer batch determinará el valor inicial real.
        self.H_c   = torch.zeros(num_classes, device=device)  # Entropía media por clase
        self.S_c   = torch.zeros(num_classes, device=device)  # Sensibilidad media por clase
        self.err_c = torch.zeros(num_classes, device=device)  # Tasa de error por clase

    @torch.no_grad()
    def update(self, y, entropy, sensitivity, logits_nat):
        """
        Actualiza las estadísticas EMA para cada clase presente en el batch.

        Parámetros:
          y           -> Etiquetas verdaderas del batch, forma [B].
          entropy     -> Entropía local por muestra,    forma [B].
          sensitivity -> Sensibilidad local por muestra, forma [B].
          logits_nat  -> Logits del modelo sobre muestras limpias, forma [B, C].
                         Se usan para calcular err_c: predicción incorrecta = error.

        Para cada clase c presente en el batch:
          1. Filtra las muestras cuya etiqueta sea c  -> subconjunto B_c
          2. Calcula el promedio de entropía, sensibilidad y error sobre B_c
          3. Aplica EMA: v_c = (1 - rho) * v_c + rho * promedio_batch
        """
        # Predicciones del modelo sobre las muestras limpias
        preds = logits_nat.argmax(dim=1)  # [B]

        for c in range(self.num_classes):
            # Máscara booleana: muestras del batch que pertenecen a la clase c
            mask = (y == c)

            # Si no hay muestras de la clase c en este batch, no actualizar
            if mask.sum() == 0:
                continue

            # --- Promedio de entropía del batch para la clase c ---
            H_batch = entropy[mask].mean()

            # --- Promedio de sensibilidad del batch para la clase c ---
            S_batch = sensitivity[mask].mean()

            # --- Tasa de error del batch para la clase c ---
            # err_c^(t) = 1 - (predicciones correctas de clase c / total muestras de clase c)
            correct = (preds[mask] == c).float().mean()
            err_batch = 1.0 - correct

            # --- Actualización EMA ---
            self.H_c[c]   = (1 - self.rho) * self.H_c[c]   + self.rho * H_batch
            self.S_c[c]   = (1 - self.rho) * self.S_c[c]   + self.rho * S_batch
            self.err_c[c] = (1 - self.rho) * self.err_c[c] + self.rho * err_batch

    @torch.no_grad()
    def get_alpha_beta(self, y, alpha_base: float, beta_base: float):
        """
        Calcula alpha_c y beta_c adaptativos para cada muestra del batch,
        usando las estadísticas globales normalizadas de su clase.

        Fórmulas (Opción B — con piso garantizado):
          alpha_c = alpha_base * (1 + H_c_tilde)   (escala en [alpha_base, 2*alpha_base])
          beta_c  = beta_base  * (1 + S_c_tilde)   (escala en [beta_base,  2*beta_base])

        Donde H_c_tilde y S_c_tilde son versiones min-max normalizadas
        de H_c y S_c sobre todas las clases (rango [0, 1]).

        A diferencia de la Opción A (alpha_c = alpha_base * H_tilde_c), aquí
        ninguna clase puede colapsar a cero: el factor (1 + H_tilde_c) siempre
        vale al menos 1.0. Esto evita que clases con baja incertidumbre relativa
        reciban un lambda efectivamente nulo y queden sin defensa adversarial.

        La clase con mayor H_c recibe alpha_c = 2 * alpha_base.
        La clase con menor H_c recibe alpha_c = alpha_base (no cero).
        Análogo para beta_c con S_c.

        Parámetros:
          y          -> Etiquetas verdaderas del batch, forma [B].
          alpha_base -> Hiperparámetro base para el peso de entropía.
          beta_base  -> Hiperparámetro base para el peso de sensibilidad.

        Retorna:
          alpha_per_sample -> Tensor [B] con alpha_c para cada muestra.
          beta_per_sample  -> Tensor [B] con beta_c para cada muestra.
        """
        # Normalizar estadísticas de clase al rango [0, 1] con min-max entre clases
        # H_tilde_c = (H_c - min_j H_j) / (max_j H_j - min_j H_j + eps)
        H_tilde = _normalize_class_stats(self.H_c)    # [num_classes], rango [0, 1]
        S_tilde = _normalize_class_stats(self.S_c)    # [num_classes], rango [0, 1]

        # Opción B: piso garantizado en alpha_base y beta_base.
        # El (1 + ...) asegura que ninguna clase recibe peso cero,
        # independientemente de cuán baja sea su incertidumbre o sensibilidad relativa.
        alpha_c = alpha_base * (0.5 + H_tilde)   # [num_classes], rango [alpha_base, 2*alpha_base]
        beta_c  = beta_base  * (0.5 + S_tilde)   # [num_classes], rango [beta_base,  2*beta_base]

        # Asignar a cada muestra el valor de su clase correspondiente
        alpha_per_sample = alpha_c[y]   # [B]
        beta_per_sample  = beta_c[y]    # [B]

        return alpha_per_sample, beta_per_sample

    @torch.no_grad()
    def get_alpha_beta_class(self, alpha_base: float, beta_base: float):
        """
        Versión por CLASE (no por muestra del batch) de get_alpha_beta().

        Devuelve listas Python de num_classes valores, usando las estadísticas
        EMA acumuladas hasta este momento. Se llama desde compute_loss() para
        registrar en Metrics los valores actuales *por clase*, independientemente
        del batch que se esté procesando.

        Retorna:
          alpha_class_list : list[float] de longitud num_classes
          beta_class_list  : list[float] de longitud num_classes
        """
        # Misma fórmula Opción B que get_alpha_beta(), pero devuelve listas Python
        # para que Metrics pueda serializarlas directamente al CSV.
        H_tilde = _normalize_class_stats(self.H_c)             # [num_classes]
        S_tilde = _normalize_class_stats(self.S_c)             # [num_classes]
        alpha_c = (alpha_base * (1.0 + H_tilde)).tolist()      # list[float]
        beta_c  = (beta_base  * (1.0 + S_tilde)).tolist()      # list[float]
        return alpha_c, beta_c


# ===========================================================================
# Función de pérdida D-TRADES
# ===========================================================================

def d_trades_loss(
    model,
    x_natural,
    y,
    optimizer,
    class_stats,                  # Instancia de ClassStats con estadísticas EMA por clase
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    distance='l_inf',
    alpha_base=1.0,               # Peso base de entropía (antes: alpha)
    beta_base=1.0,                # Peso base de sensibilidad (antes: beta)
    gamma=0.5,                    # Peso del término de error de clase en lambda
    normalize_terms=True,         # True: normaliza entropía y sensibilidad locales a [0,1]
    per_sample_sensitivity=True,  # True: cálculo exacto por muestra (loop); False: aproximación por batch
    beta_trades=6.0,
    EPS=1e-12,                    # Estabilidad numérica para log(0) en entropía
):
    """
    Calcula la pérdida D-TRADES con lambda dinámico adaptativo por clase.

    La función de pérdida es:
        L_D-TRADES(f, x, y) = L_CE(f(x), y) + lambda(x) * KL(f(x) || f(x+delta))

    Donde lambda(x) es dinámico y depende de cada muestra:
        lambda(x) = alpha_c * H(x) + beta_c * S(x) + gamma * err_c

    Con:
        H(x)    -> Entropía local de la muestra (incertidumbre inmediata)
        S(x)    -> Sensibilidad adversarial local normalizada (norma del gradiente KL)
        err_c   -> Error acumulado EMA de la clase c (memoria global)
        alpha_c -> alpha_base * (1 + H_tilde_c)  (Opción B: piso en alpha_base, techo en 2*alpha_base)
        beta_c  -> beta_base  * (1 + S_tilde_c)  (Opción B: piso en beta_base,  techo en 2*beta_base)
        gamma   -> Hiperparámetro fijo que controla el peso del error de clase

    Parámetros:
      model          -> Red neuronal CNN a entrenar.
      x_natural      -> Batch de imágenes limpias, forma [B, C, H, W].
      y              -> Etiquetas verdaderas, forma [B].
      optimizer      -> Optimizador de PyTorch (usado externamente, no dentro de esta función).
      class_stats    -> Instancia de ClassStats con las estadísticas EMA actualizadas.
      step_size      -> Tamaño de paso del ataque PGD.
      epsilon        -> Radio máximo de perturbación adversarial.
      perturb_steps  -> Número de pasos del ataque PGD.
      distance       -> Norma del ataque: 'l_inf' o 'l_2'.
      alpha_base     -> Peso base para la entropía en lambda.
      beta_base      -> Peso base para la sensibilidad en lambda.
      gamma          -> Peso del error de clase en lambda.
      normalize_terms-> Si True, normaliza entropía y sensibilidad locales a [0,1].
      per_sample_sensitivity -> Si True, calcula la sensibilidad exacta por muestra (más lento).
      EPS            -> Constante numérica para evitar log(0) en la entropía.

    Retorna:
      loss_total          -> Pérdida total D-TRADES (escalar).
      lam                 -> Vector lambda dinámico por muestra [B] (detached).
      loss_natural        -> Pérdida de clasificación limpia (detached, para logging).
      loss_robust_dynamic -> Pérdida robusta ponderada (detached, para logging).
    """

    # ---------- Ataque PGD (igual que TRADES) ----------
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

    # ---------- Forward limpio y pérdidas base ----------

    # Salida del modelo para las imágenes limpias
    logits_nat = model(x_natural)

    # Salida del modelo para las imágenes adversariales
    logits_adv = model(x_adv.detach())

    # Probabilidades limpias (softmax) y log-probabilidades adversariales
    probs_nat    = F.softmax(logits_nat, dim=1).clamp(min=1e-8, max=1.0)
    log_probs_adv = F.log_softmax(logits_adv, dim=1).clamp(min=-50, max=0)

    # Pérdida de clasificación estándar sobre datos limpios
    loss_natural = F.cross_entropy(logits_nat, y, reduction='mean')

    # KL por muestra: KL(f(x) || f(x+delta)), forma [B]
    # Representa qué tanto cambia la predicción del modelo al agregar la perturbación.
    kl_per_example = F.kl_div(
        log_probs_adv,
        probs_nat, 
        reduction='none'
    ).sum(dim=1)

    # ---------- Cálculo de entropía local H(x) ----------
    # H(x) = - sum_c p_c * log(p_c)
    # Captura la incertidumbre inmediata del modelo sobre la muestra x.
    # Se clampea probs_nat con EPS para evitar log(0) -> inf.
    # Resultado: tensor [B], uno por muestra.
    entropy = -(probs_nat * torch.log(probs_nat.clamp_min(EPS))).sum(dim=1)

    # ---------- Cálculo de sensibilidad local S(x) ----------
    # S(x) = || nabla_{x+delta} KL(f(x) || f(x+delta)) ||_2
    # Captura cuánto cambia la KL si se perturba levemente x+delta.
    # Un valor alto indica que la muestra está en una zona muy sensible.
    if per_sample_sensitivity:
        # Versión exacta por muestra (loop): más lenta pero precisa.
        # Para cada muestra i:
        #   1. Clona x_adv[i] con gradiente activo.
        #   2. Calcula KL_i individual.
        #   3. Obtiene el gradiente ∂KL_i/∂x_adv[i] via autograd.
        #   4. Calcula la norma L2 del gradiente aplanado.
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
        # Versión por batch (aproximación): un único grad para la KL promedio.
        # Más rápida pero menos precisa por muestra.
        x_adv_req = x_adv.detach().clone().requires_grad_(True)
        kl_total = F.kl_div(
            F.log_softmax(model(x_adv_req), dim=1),
            probs_nat.detach(),
            reduction='batchmean'
        )
        g = torch.autograd.grad(kl_total, x_adv_req, create_graph=False)[0]
        sensitivity = g.view(batch_size, -1).norm(p=2, dim=1)   # [B]
        sensitivity = torch.log1p(sensitivity)

    # ---------- Normalización local de entropía y sensibilidad ----------
    # Se normalizan al rango [0, 1] usando min-max sobre el batch actual.
    # Esto evita que la sensibilidad, que puede tomar valores grandes,
    # domine sobre la entropía, que ya está acotada en [0, log(C)].
    # La normalización es local (por batch), no global.
    # Normalización más estable (NO dependiente del batch)

    num_classes = probs_nat.size(1)
    
    entropy_n = entropy / torch.log(torch.tensor(num_classes, device=entropy.device))
    sensitivity_log = torch.log1p(sensitivity)
    #entropy_n = entropy / torch.log(torch.tensor(num_classes, device=entropy.device))
    sensitivity_n = sensitivity_log / (sensitivity_log.mean() + 1e-8)

    #if normalize_terms:
    #    entropy_n    = _normalize_batch(entropy)      # H(x) normalizado, [B]
    #    sensitivity_n = _normalize_batch(sensitivity) # S(x) normalizado, [B]
    #else:
    #    entropy_n    = entropy
    #    sensitivity_n = sensitivity

    # ---------- Actualización de estadísticas EMA por clase ----------
    # Se actualizan H_c, S_c y err_c usando los valores locales del batch actual.
    # Se usan los valores sin normalizar (entropy, sensitivity) para las estadísticas
    # globales, ya que la normalización local depende del batch y cambiaría cada vez.
    # La normalización de H_c y S_c para alpha_c y beta_c se hace dentro de ClassStats.
    class_stats.update(y, entropy.detach(), sensitivity.detach(), logits_nat.detach())

    # ---------- Cálculo de alpha_c y beta_c por clase ----------
    # alpha_c = alpha_base * H_tilde_c
    # beta_c  = beta_base  * S_tilde_c
    # Donde H_tilde_c y S_tilde_c son las estadísticas EMA normalizadas
    # con min-max entre clases (no entre muestras del batch).
    # Resultado: tensores [B], uno por muestra según su clase y.
    alpha_per_sample, beta_per_sample = class_stats.get_alpha_beta(
        y, alpha_base, beta_base
    )

    # ---------- Construcción de lambda dinámico ----------
    # lambda(x) = alpha_c * H(x) + beta_c * S(x) + gamma * err_c
    #
    # Tres niveles de información:
    #   alpha_c * H(x)   -> incertidumbre local modulada por la clase
    #   beta_c  * S(x)   -> sensibilidad local modulada por la clase
    #   gamma   * err_c  -> error global acumulado de la clase (inspirado en MART)
    #
    # Se detacha lambda para que no entre en el grafo de backpropagation.
    # Lambda actúa como peso del término KL, no como parte del modelo.
    err_per_sample = class_stats.err_c[y]   # [B], error EMA de la clase de cada muestra

    lam = (
        alpha_per_sample * entropy_n
        + beta_per_sample  * sensitivity_n
        + gamma            * err_per_sample
    ).detach()   # [B]

    lam = lam /(lam.mean() + 1e-8)
    
    kl_mean = kl_per_example.mean().detach()
    lam = lam * kl_mean
    #lam = lam * beta_trades
    
    #confidence = probs_nat.max(dim=1)[0]
    #difficulty = 1.0 - confidence
    
    lam = lam.clamp(0.1, 6.0)
    # ---------- Pérdida robusta ponderada dinámicamente ----------
    # L_RD = mean( lambda(x_i) * KL(f(x_i) || f(x_i + delta)) )
    #loss_robust_dynamic = (lam * kl_per_example).mean() * beta_trades
    loss_robust_dynamic = (lam * kl_per_example).mean()

    # Pérdida total D-TRADES
    # L_D-TRADES = L_CE(f(x), y) + L_RD
    loss_total = loss_natural + loss_robust_dynamic

    return loss_total, lam, loss_natural.detach(), loss_robust_dynamic.detach()
