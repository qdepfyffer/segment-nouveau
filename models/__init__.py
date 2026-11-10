"""
ADDING A NEW MODEL:
If you want to add a model, you'll need to do a few things: Create an implementation file for the model you want to add
and add its name to the MODELS dictionary below. For an example (very simple) implementation file, see deeplabv3.py. In
short, the model's forward function must take a batch of images and return the logits with the shape [batch, classes, height, width].
New models must also declare a size_multiple (use 32 if you're unsure), and must call pad_to_multiple in its forward function
After creating the wrapper you need to also create a <model name>.yaml file under conf/model/, with at least `name: <model name>`, any
extra keys in the model's YAML become constructor arguments. After adding and training a new model, run
`python -m tools.check_sizing checkpoint=<run> crop=<crop>` for each crop you plan to report
"""

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from models.deeplabv3 import DeepLabV3MobileNet
from models.fcdensenet import FCDenseNet103
from models.frrn_a import FRRNA
from models.frrn_a_lite import FRRNALite

MODELS = {
    "deeplabv3": DeepLabV3MobileNet,
    "frrn_a": FRRNA,
    "frrn_a_lite": FRRNALite,
    "fcdensenet103": FCDenseNet103,
}


def build_model(model_cfg: DictConfig, num_classes: int):
    """
    Returns the model corresponding to the specified name
    :param model_cfg:   Model section of config, name plus any options
    :param num_classes: The number of classes the model should be initialized with
    :return:            The built model
    """

    # Load options and get name
    options = cast(dict[str, Any], OmegaConf.to_container(model_cfg, resolve=True))  # cast here shuts up type checker, it'll never be anything other than a dict with string keys.
    name = options.pop("name")

    if name not in MODELS:
        raise ValueError(
            f"Model {name} not found in models list. Available models: {sorted(MODELS)}"
        )

    return MODELS[name](num_classes=num_classes, **options)
