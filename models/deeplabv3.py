from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from models.padding import pad_to_multiple


class DeepLabV3MobileNet(nn.Module):
    """
    Wraps the DeepLabV3 model for use in this training suite
    """

    size_multiple = 16  # Every model needs this, use 32 as a reasonable guess.
    has_pretrained_backbone = True  # Lets build_model() skip downloading the backbone if the weights are going to be replaced by a checkpoint anyway

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.net = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,  # `pretrained` is true when we want to download the weights and false when we don't so None ends up giving us random weights with no download
            num_classes=num_classes
        )


    def forward(self, images):
        """
        Defines the forward pass of the model
        :param images:  The images being shown to the model
        :return:        Logits of shape [batch_size, num_classes, height, width]
        """
        height, width = images.shape[-2:]
        padded = pad_to_multiple(images, self.size_multiple)
        output = self.net(padded)
        logits = output["out"]
        return logits[..., :height, :width]
