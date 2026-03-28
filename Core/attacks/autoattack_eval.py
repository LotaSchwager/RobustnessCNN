import torch

def autoattack_eval(model, device, test_loader, eps=0.031, max_samples=1000):
    from autoattack import AutoAttack

    model.eval()
    model.to(device)

    x_test = []
    y_test = []
    total = 0

    for x, y in test_loader:
        x_test.append(x)
        y_test.append(y)
        total += x.size(0)
        if total >= max_samples:
            break

    x_test = torch.cat(x_test)[:max_samples].to(device)
    y_test = torch.cat(y_test)[:max_samples].to(device)

    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=eps,
        version='standard',
        device=device
    )

    x_adv = adversary.run_standard_evaluation(x_test, y_test)

    with torch.no_grad():
        pred_nat = model(x_test).argmax(1)
        pred_adv = model(x_adv).argmax(1)

    natural_acc = (pred_nat == y_test).float().mean().item()
    robust_acc  = (pred_adv == y_test).float().mean().item()

    return natural_acc, robust_acc
