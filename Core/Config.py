import os
import torch
import time

class Config:
    def __init__(
            self, 
            batch, 
            epochs, 
            weight_decay, 
            lr, 
            momentum, 
            epsilon, 
            num_steps, 
            step_size,
            save_freq,
            run_name,
            beta = 1.0,
            alpha = 1.0,
            seed = 1,
            cuda = True,
            results_dir = '../Resultado/modelo',
            temp_dir = '../Resultado/Temp',
        ):
        self._alpha = alpha
        self._beta = beta
        self._batch = batch
        self._epochs = epochs
        self._weight_decay = weight_decay
        self._lr = lr
        self._momentum = momentum
        self._epsilon = epsilon
        self._num_steps = num_steps
        self._step_size = step_size
        self._save_freq = save_freq
        self._run_name = run_name
        self._seed = seed
        self._run_id = time.strftime("%Y%m%d-%H%M%S")
        self._results_dir = results_dir + f'/{self._run_id}'
        self._temp_dir = temp_dir + f'/{self._run_id}'

        # Crear directorios
        os.makedirs(self._results_dir, exist_ok=True)
        os.makedirs(self._temp_dir, exist_ok=True)

        # Configuración de CUDA
        self._use_cuda = bool(cuda) and torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        self._kwargs = {"num_workers": 4, "pin_memory": True} if self._use_cuda else {}

        # Seeds
        torch.manual_seed(self._seed)
        if self._use_cuda:
            torch.cuda.manual_seed_all(self._seed)
    
    # Guarda los checkpoints
    def save_checkpoints(self, epoch, optimizer, model):
        if epoch % self._save_freq == 0 and epoch < self._epochs:
            torch.save(model.state_dict(), os.path.join(self._temp_dir, f"{self._run_name}_checkpoint_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(self._temp_dir, f"{self._run_name}_checkpoint_{epoch}.tar"))
        elif epoch == self._epochs:
            torch.save(model.state_dict(), os.path.join(self._results_dir, f"{self._run_name}_final.pt"))
            torch.save(optimizer.state_dict(), os.path.join(self._results_dir, f"{self._run_name}_final.tar"))
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def beta(self):
        return self._beta
    
    @property
    def batch(self):
        return self._batch
    
    @property
    def epochs(self):
        return self._epochs
    
    @property
    def weight_decay(self):
        return self._weight_decay
    
    @property
    def lr(self):
        return self._lr
    
    @property
    def momentum(self):
        return self._momentum
    
    @property
    def epsilon(self):
        return self._epsilon
    
    @property
    def num_steps(self):
        return self._num_steps
    
    @property
    def step_size(self):
        return self._step_size
    
    @property
    def save_freq(self):
        return self._save_freq
    
    @property
    def run_name(self):
        return self._run_name
    
    @property
    def seed(self):
        return self._seed
    
    @property
    def run_id(self):
        return self._run_id
    
    @property
    def results_dir(self):
        return self._results_dir
    
    @property
    def temp_dir(self):
        return self._temp_dir
    
    @property
    def device(self):
        return self._device
    
    @property
    def kwargs(self):
        return self._kwargs
    
    @property
    def use_cuda(self):
        return self._use_cuda