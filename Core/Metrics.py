import os

class Metrics:
    def __init__(self, results_dir: str):
        self._lambda_min = []
        self._lambda_max = []
        self._lambda_mean = []
        self._loss_natural = []
        self._loss_robust = []
        self._loss = []
        self._robust_acc = []
        self._natural_acc = []
        self._robust_drop = []
        self._attack_success_rate = []
        # Obtener directorio para metrica basada en la hora actual
        self._results_dir = results_dir
        if not os.path.exists(self._results_dir):
            os.makedirs(self._results_dir)
    
    def update(self, lambda_min, lambda_max, lambda_mean, loss_natural, loss_robust, loss, acc_natural, acc_robust, robust_drop, attack_success_rate):
        self._lambda_min.append(lambda_min)
        self._lambda_max.append(lambda_max)
        self._lambda_mean.append(lambda_mean)
        self._loss_natural.append(loss_natural)
        self._loss_robust.append(loss_robust)
        self._loss.append(loss)
        self._robust_acc.append(acc_robust)
        self._natural_acc.append(acc_natural)
        self._robust_drop.append(robust_drop)
        self._attack_success_rate.append(attack_success_rate)
    
    # Guarda las metricas en formato CSV
    def save_metrics(self):
        with open(os.path.join(self._results_dir, 'metricas.csv'), 'w') as f:
            f.write('lambda_min,lambda_max,lambda_mean,loss_natural,loss_robust,loss,acc_natural,acc_robust,robust_drop,attack_success_rate\n')
            for i in range(len(self._lambda_min)):
                f.write(f'{self._lambda_min[i]},{self._lambda_max[i]},{self._lambda_mean[i]},{self._loss_natural[i]},{self._loss_robust[i]},{self._loss[i]},{self._natural_acc[i]},{self._robust_acc[i]},{self._robust_drop[i]},{self._attack_success_rate[i]}\n')
    
    @property
    def lambda_min(self):
        return self._lambda_min
    
    @property
    def lambda_max(self):
        return self._lambda_max
    
    @property
    def lambda_mean(self):
        return self._lambda_mean
    
    @property
    def loss_natural(self):
        return self._loss_natural
    
    @property
    def loss_robust(self):
        return self._loss_robust
    
    @property
    def loss(self):
        return self._loss
    
    @property
    def robust_acc(self):
        return self._robust_acc
    
    @property
    def natural_acc(self):
        return self._natural_acc
    
    @property
    def robust_drop(self):
        return self._robust_drop
    
    @property
    def attack_success_rate(self):
        return self._attack_success_rate