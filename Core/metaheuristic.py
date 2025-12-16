import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Models.resnet import *
from Metodo.dtrades import *
from Models.normalize import NormalizeLayer
from Config import Config
from eval import eval_adv_test_whitebox_pgd as eval_adv_test_whitebox

class Metaheuristic:
    def __init__(self, model_factory, train_loader, test_loader, cfg: Config):
        self.cfg = cfg
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_factory = model_factory 

        self.epochs = 5
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std  = (0.2470, 0.2435, 0.2616)

    def run_candidate(self, alpha: float, beta: float):
        base = self.model_factory()
        model = nn.Sequential(NormalizeLayer(self.mean, self.std), base).to(self.cfg.device)

        optimizer = optim.SGD(
            model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay
        )

        losses = []
        for epoch in range(1, self.epochs + 1):
            loss = train(self.cfg, model, self.cfg.device, self.train_loader, optimizer, epoch, alpha, beta)
            losses.append(loss)

        eval_stats = eval_adv_test_whitebox(
            model, 
            self.cfg.device, 
            self.test_loader,
            self.cfg.epsilon,
            self.cfg.num_steps,
            self.cfg.step_size,
        )

        NA = eval_stats["natural_acc"]
        RA = eval_stats["robust_acc"] 
        ok = (NA >= 0.85 * RA)
        return {"alpha": alpha, "beta": beta, "NA": float(NA), "RA": float(RA), "losses": losses, "ok": ok}

    # Elegir el mejor candidato
    def pick_best(self, results):
        ok = [r for r in results if r['ok']]
        if ok:
            return max(ok, key=lambda x: x['RA'])
        
        return max(results, key=lambda r: r["RA"])

    # Vecinos
    def neighbor(self, alpha0: float, beta0: float, *, radius: float = 0.2, step: float = 0.1, lo: float = 0.0, hi: float = 3.0):
        """
        Búsqueda local alrededor de (alpha0, beta0)
        """
        alphas = np.arange(alpha0 - radius, alpha0 + radius + 1e-9, step)
        betas  = np.arange(beta0  - radius, beta0  + radius + 1e-9, step)

        # clamp al dominio válido
        alphas = [round(min(hi, max(lo, a)), 4) for a in alphas]
        betas  = [round(min(hi, max(lo, b)), 4) for b in betas]

        candidates = set((a, b) for a in alphas for b in betas)

        results = []
        for a, b in sorted(candidates):
            print(f"[META][NEIGHBOR] alpha={a}, beta={b}")
            results.append(self.run_candidate(a, b))

        best = self.pick_best(results)
        return best

    # Mejor alpha beta
    def alpha_beta_search(self):
        base_vals = [0,1,2,3]
        results = []
        for a in base_vals:
            for b in base_vals:
                print(f"[META] Testing alpha={a}, beta={b}")
                res = self.run_candidate(a, b)
                results.append(res)

        best = self.pick_best(results)
        print(f"[META] Best: alpha={best['alpha']}, beta={best['beta']}, NA={best['NA']}, RA={best['RA']}")

        best_local = self.neighbor(best['alpha'], best['beta'])
        print(f"[META] Best local: alpha={best_local['alpha']}, beta={best_local['beta']}, NA={best_local['NA']}, RA={best_local['RA']}")
        return best_local

    # Inicio de ejecución
    def main(self):
        best = self.alpha_beta_search()
        alpha, beta = best["alpha"], best["beta"]
        return alpha, beta