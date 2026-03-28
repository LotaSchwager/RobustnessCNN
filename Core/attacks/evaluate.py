import torch
from .pgd import pgd_attack
from .fgsm import fgsm_attack
from .autoattack_eval import autoattack_eval

def evaluate_all_attacks(model, device, test_loader, cfg):
    model.eval()
    model.to(device)

    natural_correct = 0
    pgd_correct = 0
    fgsm_correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        total += data.size(0)

        # -------------------------
        # Natural
        # -------------------------
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(1)
            natural_correct += pred.eq(target).sum().item()

        # -------------------------
        # PGD
        # -------------------------
        X_pgd = pgd_attack(
            model, data, target,
            epsilon=cfg.epsilon,
            step_size=cfg.step_size,
            num_steps=20
        )

        with torch.no_grad():
            pred = model(X_pgd).argmax(1)
            pgd_correct += pred.eq(target).sum().item()

        # -------------------------
        # FGSM
        # -------------------------
        X_fgsm = fgsm_attack(model, data, target, epsilon=cfg.epsilon)

        with torch.no_grad():
            pred = model(X_fgsm).argmax(1)
            fgsm_correct += pred.eq(target).sum().item()

    natural_acc = natural_correct / total
    pgd_acc     = pgd_correct / total
    fgsm_acc    = fgsm_correct / total

    # -------------------------
    # AutoAttack
    # -------------------------
    try:
        auto_nat, auto_rob = autoattack_eval(model, device, test_loader, eps=cfg.epsilon)
    except Exception as e:
        print("[WARNING] AutoAttack no disponible:", e)
        auto_nat, auto_rob = None, None

    print("\n===== RESULTADOS =====")
    print(f"Natural Acc : {natural_acc:.4f}")
    print(f"PGD Acc     : {pgd_acc:.4f}")
    print(f"FGSM Acc    : {fgsm_acc:.4f}")
    if auto_nat is not None:
        print(f"AutoAttack  : {auto_rob:.4f}")

    return {
        "natural": natural_acc,
        "pgd": pgd_acc,
        "fgsm": fgsm_acc,
        "auto": auto_rob
    }
