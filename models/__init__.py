from models.deeplabv3 import DeepLabV3MobileNet

MODELS = {
    "deeplabv3": DeepLabV3MobileNet
}

def build_model(name: str, num_classes: int):
    if not name in MODELS:
        raise ValueError(f"Model {name} not found in models list. Available models: {sorted(MODELS)}")

    return MODELS[name](num_classes=num_classes)