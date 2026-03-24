"""
Registro centralizado de métodos de defensa adversarial.

Para añadir un método nuevo:
  1. Crea Metodo/mi_metodo.py implementando las 4 funciones con la misma firma:
       - compute_loss(model, x, y, cfg, method_state) -> (loss_tensor, info_dict)
       - make_state(cfg, device)                      -> estado mutable del método
       - save_state(state, base_path)                 -> serializa estado a disco
       - load_state(state, base_path, device)         -> restaura estado desde disco
  2. Regístralo en METHODS usando import relativo:
       from . import mi_metodo
       METHODS["mi_metodo"] = mi_metodo
  3. Listo — main.py y train.py lo usarán sin modificación alguna.
"""

from . import dtrades as _dtrades

METHODS: dict = {
    "d_trades": _dtrades,
}


def get_method(name: str) -> dict:
    """
    Devuelve el módulo del método indicado como dict de funciones:
        {
          "compute_loss": fn,
          "make_state":   fn,
          "save_state":   fn,
          "load_state":   fn,
        }
    """
    if name not in METHODS:
        raise ValueError(
            f"Método '{name}' no registrado en Metodo/__init__.py. "
            f"Disponibles: {list(METHODS.keys())}"
        )
    module = METHODS[name]
    return {
        "compute_loss": module.compute_loss,
        "make_state":   module.make_state,
        "save_state":   module.save_state,
        "load_state":   module.load_state,
    }
