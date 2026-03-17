import os
import torch
import time

class Config:
    def __init__(
            self, 
            batch=128, 
            epochs=100, 
            weight_decay=5e-4, 
            lr=0.1, 
            momentum=0.9, 
            epsilon=8/255, 
            num_steps=10, 
            step_size=2/255,
            save_freq=5,
            log_interval=100,
            test_batch=256,
            run_name="dtrades_run",
            beta=1.0,
            alpha=1.0,
            seed=1,
            cuda=True,
            results_dir='./Resultado/modelo',
            temp_dir='./Resultado/Temp',
        ):
        self._alpha = alpha
        self._beta = beta
        self._batch = batch
        self._test_batch = test_batch
        self._epochs = epochs
        self._weight_decay = weight_decay
        self._lr = lr
        self._momentum = momentum
        self._epsilon = epsilon
        self._num_steps = num_steps
        self._step_size = step_size
        self._save_freq = save_freq
        self._log_interval = log_interval
        self._run_name = run_name
        self._seed = seed
        self._run_id = os.getenv("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
        
        # Ajustar directorios relativos a la raíz si es necesario, 
        # pero por ahora mantenemos la lógica del usuario
        self._results_dir = os.path.join(results_dir, self._run_id)
        self._temp_dir = os.path.join(temp_dir, self._run_id)

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
        # Guarda en Temp cada multiplo de save_freq (excepto la útlima época)
        if epoch % self._save_freq == 0 and epoch < self._epochs:
            torch.save(
                model.state_dict(),
                os.path.join(self._temp_dir, f"{self._run_name}_checkpoint_{epoch}.pt")
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(self._temp_dir, f"{self._run_name}_checkpoint_{epoch}.tar")
            )
            print(f"[CKPT] Checkpoint temporal guardado en época {epoch}.")
        # Al finalizar el entrenamiento, guarda el modelo definitivo en Resultado/modelo
        if epoch == self._epochs:
            torch.save(
                model.state_dict(),
                os.path.join(self._results_dir, f"{self._run_name}_final.pt")
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(self._results_dir, f"{self._run_name}_final.tar")
            )
            print(f"[CKPT] Modelo final guardado en: {self._results_dir}")
    
    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = float(value)

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = float(value)
    
    @property
    def batch(self): return self._batch

    @property
    def test_batch(self): return self._test_batch
    
    @property
    def epochs(self): return self._epochs

    @epochs.setter
    def epochs(self, value: int) -> None:
        self._epochs = int(value)
    
    @property
    def weight_decay(self): return self._weight_decay
    
    @property
    def lr(self): return self._lr
    
    @property
    def momentum(self): return self._momentum
    
    @property
    def epsilon(self): return self._epsilon
    
    @property
    def num_steps(self): return self._num_steps

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        self._num_steps = int(value)
    
    @property
    def step_size(self): return self._step_size
    
    @property
    def save_freq(self): return self._save_freq

    @property
    def log_interval(self): return self._log_interval
    
    @property
    def run_name(self): return self._run_name
    
    @property
    def seed(self): return self._seed
    
    @property
    def run_id(self): return self._run_id
    
    @property
    def results_dir(self): return self._results_dir
    
    @property
    def temp_dir(self): return self._temp_dir
    
    @property
    def device(self): return self._device
    
    @property
    def kwargs(self): return self._kwargs
    
    @property
    def use_cuda(self): return self._use_cuda