import torch


def autoattack_eval(
    model,
    device,
    test_loader,
    eps=0.031,
    max_samples=1000
):

    from autoattack import AutoAttack

    model.eval()
    model.to(device)

    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=eps,
        version='custom',
        device=device
    )

    # SOLO ataques livianos/recomendados
    adversary.attacks_to_run = [
        'apgd-ce',
        'square'
    ]

    total_correct_nat = 0
    total_correct_adv = 0
    total_seen = 0

    for x, y in test_loader:

        # Limitar samples totales
        if total_seen >= max_samples:
            break

        x = x.to(device)
        y = y.to(device)

        current_bs = x.size(0)

        # -------------------------------------------------
        # NATURAL
        # -------------------------------------------------
        with torch.no_grad():
            pred_nat = model(x).argmax(1)

        total_correct_nat += (pred_nat == y).sum().item()

        # -------------------------------------------------
        # AUTOATTACK
        # -------------------------------------------------
        x_adv = adversary.run_standard_evaluation(
            x,
            y,
            bs=1
        )

        with torch.no_grad():
            pred_adv = model(x_adv).argmax(1)

        total_correct_adv += (pred_adv == y).sum().item()

        total_seen += current_bs

        # -------------------------------------------------
        # LIMPIEZA GPU
        # -------------------------------------------------
        del x_adv

        torch.cuda.empty_cache()

    natural_acc = total_correct_nat / total_seen
    robust_acc = total_correct_adv / total_seen

    return natural_acc, robust_acc

# def autoattack_eval(model, device, test_loader, eps=0.031, max_samples=1000):
#     from autoattack import AutoAttack
#
#     model.eval()
#     model.to(device)
#
#     x_test = []
#     y_test = []
#     total = 0
#
#     for x, y in test_loader:
#         x_test.append(x)
#         y_test.append(y)
#         total += x.size(0)
#         if total >= max_samples:
#             break
#
#     x_test = torch.cat(x_test)[:max_samples].to(device)
#     y_test = torch.cat(y_test)[:max_samples].to(device)
#
#     adversary = AutoAttack(
#         model,
#         norm='Linf',
#         eps=eps,
#         version='standard',
#         device=device
#     )
#
#     x_adv = adversary.run_standard_evaluation(x_test, y_test)
#
#     with torch.no_grad():
#         pred_nat = model(x_test).argmax(1)
#         pred_adv = model(x_adv).argmax(1)
#
#     natural_acc = (pred_nat == y_test).float().mean().item()
#     robust_acc  = (pred_adv == y_test).float().mean().item()
#
#     return natural_acc, robust_acc
