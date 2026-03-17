from .resnet import ResNet18, ResNet50
from .vgg import VGG

def get_model(name: str, num_classes: int = 10):
    """
    Fábrica de modelos centralizada. 
    Para añadir tu propio modelo, simplemente impórtalo aquí 
    y añade una condición en el buscador.
    """
    name_lower = name.lower()
    
    # --- Modelos Estándar ---
    if name_lower == 'resnet18':
        return ResNet18(num_classes=num_classes)
    if name_lower == 'resnet50':
        return ResNet50(num_classes=num_classes)
    if name_lower.startswith('vgg'):
        vgg_type = name.upper() # Ej: VGG16
        return VGG(vgg_type, num_classes=num_classes)
    
    # --- Aquí puedes añadir tu modelo personalizado ---
    # if name_lower == 'mi_modelo_propio':
    #     from .mi_archivo import MiModelo
    #     return MiModelo(num_classes=num_classes)

    raise ValueError(f"Modelo '{name}' no reconocido en la fábrica de Models. "
                     f"Asegúrate de registrarlo en Models/__init__.py")
