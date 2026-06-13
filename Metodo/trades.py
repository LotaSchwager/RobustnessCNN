import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TRADES(nn.Module):

    def __init__(
        self,
        step_size=0.003,
        epsilon=0.031,
        perturb_steps=10,
        beta=6.0,
        distance='l_inf'
    ):

        super(TRADES, self).__init__()

        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance

        self.criterion_kl_sum = nn.KLDivLoss(reduction='sum')

    def forward(self, model, x_natural, y):

        model.eval()

        batch_size = x_natural.size(0)

        x_adv = (
            x_natural.detach()
            + 0.001 * torch.randn_like(x_natural).detach()
        )

        with torch.no_grad():
            probs_nat_pgd = F.softmax(
                model(x_natural),
                dim=1
            )

        # =====================================================
        # PGD L_inf
        # =====================================================

        if self.distance == "l_inf":

            for _ in range(self.perturb_steps):

                x_adv.requires_grad_(True)

                with torch.enable_grad():

                    loss_kl = self.criterion_kl_sum(
                        F.log_softmax(model(x_adv), dim=1),
                        probs_nat_pgd
                    )

                grad = torch.autograd.grad(
                    loss_kl,
                    [x_adv]
                )[0]

                x_adv = (
                    x_adv.detach()
                    + self.step_size * torch.sign(grad.detach())
                )

                x_adv = torch.min(
                    torch.max(
                        x_adv,
                        x_natural - self.epsilon
                    ),
                    x_natural + self.epsilon
                )

                x_adv = torch.clamp(
                    x_adv,
                    0.0,
                    1.0
                )

        # =====================================================
        # PGD L2
        # =====================================================

        elif self.distance == "l_2":

            delta = (
                0.001 *
                torch.randn_like(x_natural).detach()
            )

            delta = Variable(
                delta.data,
                requires_grad=True
            )

            optimizer_delta = torch.optim.SGD(
                [delta],
                lr=self.epsilon / self.perturb_steps * 2
            )

            for _ in range(self.perturb_steps):

                adv = x_natural + delta

                optimizer_delta.zero_grad()

                with torch.enable_grad():

                    loss = -self.criterion_kl_sum(
                        F.log_softmax(
                            model(adv),
                            dim=1
                        ),
                        probs_nat_pgd
                    )

                loss.backward()

                grad_norms = (
                    delta.grad
                    .view(batch_size, -1)
                    .norm(p=2, dim=1)
                )

                safe = grad_norms.clone()
                safe[safe == 0] = 1.0

                delta.grad.div_(
                    safe.view(-1, 1, 1, 1)
                )

                zero_mask = (
                    grad_norms == 0
                ).view(-1, 1, 1, 1)

                delta.grad[zero_mask] = (
                    torch.randn_like(
                        delta.grad[zero_mask]
                    )
                )

                optimizer_delta.step()

                delta.data.add_(x_natural)

                delta.data.clamp_(
                    0,
                    1
                ).sub_(x_natural)

                delta.data.renorm_(
                    p=2,
                    dim=0,
                    maxnorm=self.epsilon
                )

            x_adv = Variable(
                x_natural + delta,
                requires_grad=False
            )

        else:

            x_adv = torch.clamp(
                x_adv,
                0.0,
                1.0
            )

        model.train()
# =====================================================
        # TRADES LOSS
        # =====================================================

        logits_nat = model(x_natural)

        loss_natural = F.cross_entropy(
            logits_nat,
            y,
            reduction='mean'
        )

        loss_robust = (
            1.0 / batch_size
        ) * self.criterion_kl_sum(
            F.log_softmax(
                model(x_adv.detach()),
                dim=1
            ),
            F.softmax(
                logits_nat.detach(),
                dim=1
            )
        )

        loss_total = (
            loss_natural
            + self.beta * loss_robust
        )

        # Modifica este diccionario exactamente así:
        info = {
            "loss_natural": loss_natural.item(),
            "loss_robust": loss_robust.item(),
            "loss": loss_total.item(),
            "beta": self.beta
        }
        
        return loss_total, info
def make_state(cfg, device):

    distance = getattr(
        cfg,
        "distance",
        "l_inf"
    )

    criterion = TRADES(
        step_size=cfg.step_size,
        epsilon=cfg.epsilon,
        perturb_steps=cfg.num_steps,
        beta=cfg.beta,
        distance=distance
    ).to(device)

    return criterion


def save_state(state, path):
    pass


def load_state(state, path, device):
    pass


def compute_loss(model, x, y, cfg, method_state):

    criterion = method_state

    loss_total, info = criterion(
        model,
        x,
        y
    )

    return loss_total, info
