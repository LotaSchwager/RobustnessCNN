import os
import torch
import time

class Config:
    def __init__(
            self, 
            dataset="cifar10",
            model="resnet18",
            method="d_trades",
            epochs=100,
            num_classes=10,       # resuelto desde DataConfig antes de crear Config
            seed=1,
            cuda=True,
            results_dir='./Resultado/modelo',
            temp_dir='./Resultado/Temp',
        ):
        self._dataset     = dataset
        self._model       = model
        self._method      = method
        self._epochs      = int(epochs)
        self._num_classes = int(num_classes)
        
        # Hiperparámetros generales de entrenamiento
        self._batch = 256
        self._test_batch = 256
        self._weight_decay = 5e-4
        self._lr = 0.2
        self._momentum = 0.9
        
        if method == "d_trades":
            self._epsilon   = 8/255 if dataset in ["cifar10", "cifar100"] else 0.3
            self._num_steps = 10
            self._step_size = 2/255 if dataset in ["cifar10", "cifar100"] else 0.01
            
            # Hiperparámetros del lambda dinámico adaptativo por clase:
            #   alpha_base: peso base de la entropía local H(x).
            #   beta_base : peso base de la sensibilidad local S(x).
            #   gamma     : peso del error acumulado de clase err_c.
            #   rho       : tasa de actualización EMA para las estadísticas por clase.
            self._alpha_base = 1.0
            self._beta_base  = 1.0
            self._gamma      = 1.0
            self._rho        = 0.1
        else:
            self._epsilon    = 8/255
            self._num_steps  = 10
            self._step_size  = 2/255
            self._alpha_base = 1.0
            self._beta_base  = 1.0
            self._gamma      = 0.0
            self._rho        = 0.1
            
        self._save_freq    = 20
        self._log_interval = 100
        self._run_name     = f"{method}_{dataset}_{model}"
        self._seed         = seed
        self._run_id       = os.getenv("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
        
        self._results_dir = os.path.join(results_dir, self._run_id)
        self._temp_dir    = os.path.join(temp_dir,    self._run_id)
        os.makedirs(self._results_dir, exist_ok=True)
        os.makedirs(self._temp_dir,    exist_ok=True)

        self._use_cuda = bool(cuda) and torch.cuda.is_available()
        self._device   = torch.device("cuda" if self._use_cuda else "cpu")
        self._kwargs   = {"num_workers": 4, "pin_memory": True} if self._use_cuda else {}

        torch.manual_seed(self._seed)
        if self._use_cuda:
            torch.cuda.manual_seed_all(self._seed)
    
    def save_checkpoints(self, epoch, optimizer, model):
        """
        Guarda checkpoints del modelo y optimizer.
        
        Durante el entrenamiento (cada save_freq épocas) los archivos van a
        temp_dir para no acumular en results_dir. En la época final van a
        results_dir como modelo definitivo.
        
        Retorna la ruta base del checkpoint guardado (sin extensión), para que
        train.py pueda guardar el .stats correspondiente en el mismo directorio
        y con el mismo nombre base, garantizando coherencia al restaurar.
        """
        if epoch % self._save_freq == 0 and epoch < self._epochs:
            base = os.path.join(self._temp_dir, f"{self._run_name}_checkpoint_{epoch}")
            torch.save(model.state_dict(),     base + ".pt")
            torch.save(optimizer.state_dict(), base + ".tar")
            print(f"[CKPT] Checkpoint temporal guardado en época {epoch}.")
            return base   # train.py usará esto para guardar base + ".stats"

        if epoch == self._epochs:
            base = os.path.join(self._results_dir, f"{self._run_name}_final")
            torch.save(model.state_dict(),     base + ".pt")
            torch.save(optimizer.state_dict(), base + ".tar")
            print(f"[CKPT] Modelo final guardado en: {self._results_dir}")
            return base   # train.py usará esto para guardar base + ".stats"

        return None   # épocas intermedias sin checkpoint — no hay .stats que guardar

    @property
    def num_classes(self):  return self._num_classes
    @property
    def dataset(self):     return self._dataset
    @property
    def model(self):       return self._model
    @property
    def method(self):      return self._method
    @property
    def batch(self):       return self._batch
    @property
    def test_batch(self):  return self._test_batch
    @property
    def epochs(self):      return self._epochs
    @property
    def weight_decay(self):return self._weight_decay
    @property
    def lr(self):          return self._lr
    @property
    def momentum(self):    return self._momentum
    @property
    def epsilon(self):     return self._epsilon
    @property
    def num_steps(self):   return self._num_steps
    @property
    def step_size(self):   return self._step_size
    @property
    def alpha_base(self):  return self._alpha_base
    @property
    def beta_base(self):   return self._beta_base
    @property
    def gamma(self):       return self._gamma
    @property
    def rho(self):         return self._rho
    @property
    def save_freq(self):   return self._save_freq
    @property
    def log_interval(self):return self._log_interval
    @property
    def run_name(self):    return self._run_name
    @property
    def seed(self):        return self._seed
    @property
    def run_id(self):      return self._run_id
    @property
    def results_dir(self): return self._results_dir
    @property
    def temp_dir(self):    return self._temp_dir
    @property
    def device(self):      return self._device
    @property
    def kwargs(self):      return self._kwargs
    @property
    def use_cuda(self):    return self._use_cuda
