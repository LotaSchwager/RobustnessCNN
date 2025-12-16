# Librerias
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# Funciones

"""
Normaliza a [0,1] por batch para que entropía y sensibilidad
Parámetros
* v -> Es un tensor.
* eps -> Es un valor de seguridad para evitar que la resta (v máximo - v mínimo) no sea cero.
* La idea es normalizar v utilizando una función para ello (v iesimo - v minimo / v maximo - v minimo).
"""
@torch.no_grad()
def _normalize_batch(v, eps=1e-8):
    vmin = v.min()
    vmax = v.max()
    return (v - vmin) / (vmax - vmin + eps)

# Función de perdida
def d_trades_loss(
    model,
    x_natural,
    y,
    optimizer,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    distance='l_inf',
    alpha=1.0,      # Peso de entropía
    beta=1.0,       # Peso de sensibilidad
    normalize_terms=True, # True: Para normalizar los valores de [B]; False: para usar los valores tal y como son
    per_sample_sensitivity=True,  # True: Exacto por-ejemplo (loop); False: rápido por-batch (aprox.)
    EPS=1e-12, # Número pequeño para asegurar estabilidad en el calculo de la entropía
):

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
            # renorm grad
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            safe = grad_norms.clone()
            safe[safe == 0] = 1.0
            delta.grad.div_(safe.view(-1,1,1,1))
            zero_mask = (grad_norms == 0).view(-1,1,1,1)
            delta.grad[zero_mask] = torch.randn_like(delta.grad[zero_mask])
            delta.grad.div_(grad_norms.view(-1,1,1,1))
            opt_delta.step()
            # proyección
            delta.data.add_(x_natural)
            delta.data.clamp_(0,1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)

        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    # ---------- Forward limpio y pérdidas base ----------

    # Salida del modelo para las imagenes limpias
    logits_nat = model(x_natural)

    # Salida de los ejemplos adversariales
    logits_adv = model(x_adv.detach())

    # Convierte los logits en probabilidades usando softmax
    probs_nat = F.softmax(logits_nat, dim=1)

    # Toma el algoritmo de las probabilidades adversariales
    log_probs_adv = F.log_softmax(logits_adv, dim=1)

    # Pérdida de entropía cruzada estándar para los datos limpios
    loss_natural = F.cross_entropy(logits_nat, y, reduction='mean')

    # Calcula la divergencia KL entre dos distribuciones
    # Esto es para el calculo de lambda
    # [B]
    kl_per_example = F.kl_div(
        log_probs_adv, probs_nat, reduction='none'
    ).sum(dim=1)

    # ---------- Cálculo de entropía ----------
    # probs_nat -> probabilidad de cada clase
    # La fórmula de la entropía es: - probabilidad de cada clase * logaritmo de la probabilidad.
    # El sum(dim=1) es la suma sobre las clases, para cada fila del batch.
    # Para evitar un inf dentro del logaritmo, si la probabilidad es cero, dentro del logaritmo.
    # Se usa EPS, un valor muy bajo que reemplazará la probabilidad 0.
    # Con el objetivo de evitar obtener un inf al momento de hacer log(0).
    entropy = -(probs_nat * torch.log(probs_nat.clamp_min(EPS))).sum(dim=1)

    # ---------- Cálculo de sensibilidad ----------
    # Calcula la norma de la gradiente de la KL (calculado anteriormente) respecto a la muestra adversarial
    if per_sample_sensitivity:
        # versión exacta por-ejemplo (loop) -- más lenta, pero correcta
        # Se toma una muestra x iesima con gradiente activo (requires_grad_(True)).
        # Se calcula su KL individual con reduction='sum'.
        # torch.autograd.grad devuelve la derivada parcial de L en x' (∂KL/∂x′).
        # Se aplana y calcula su norma L2 (norm(p=2, dim=1)).
        # Se acumulan todos los valores del kl_per_example llamada [B].

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
    else:
        # versión por-batch (aprox): un único grad para la KL total
        # kl_total: una KL promedio (escala batchmean).
        # autograd.grad: gradiente ∂KL_total/∂x_adv.
        # view(...).norm(p=2,dim=1): norma L2 obteniendo [B].

        x_adv_req = x_adv.detach().clone().requires_grad_(True)
        kl_total = F.kl_div(
            F.log_softmax(model(x_adv_req), dim=1),
            probs_nat.detach(),
            reduction='batchmean'
        )
        g = torch.autograd.grad(kl_total, x_adv_req, create_graph=False)[0]
        sensitivity = g.view(batch_size, -1).norm(p=2, dim=1)

    # ---------- Construcción de lambda ----------
    # Normaliza ambos vectores [B] al rango [0,1] si es que el flag es verdadero
    # Combina ambos términos: 𝜆 iésimo = 𝛼 * entropía + 𝛽 sensibilidad
    #   alpha: peso de la entropía (controla la influencia de la incertidumbre).
    #   beta: peso de la sensibilidad (controla la influencia de la vulnerabilidad adversarial).
    # lam.detach(): Hace que λ(x) se calcula fuera del gráfico de entrenamiento. Si no se detacha,
    # PyTorch intentaría retropropagar a través de las operaciones que generaron λ(x),
    # lo que distorsionaría la pérdida.

    if normalize_terms:
        entropy_n = _normalize_batch(entropy)
        sensitivity_n = _normalize_batch(sensitivity)
    else:
        entropy_n = entropy
        sensitivity_n = sensitivity

    lam = alpha * entropy_n + beta * sensitivity_n
    lam = lam.detach()

    # ---------- Pérdida robusta ponderada dinámicamente ----------
    # La pérdida adversarial dinamica que es la multiplicación de:
    # L_RD = El promedio de la sumatoria de (lambda dinamico * pérdida robusta por muestra)
    loss_robust_dynamic = (lam * kl_per_example).mean()

    # La pérdida total ahora es tal y como se hace en TRADES
    loss_total = loss_natural + loss_robust_dynamic

    # Retorna la pérdida total
    return loss_total, lam, loss_natural.detach(), loss_robust_dynamic.detach()