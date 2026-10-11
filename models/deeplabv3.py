from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class DeepLabV3MobileNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=num_classes
    )


    def forward(self, images):
        output = self.net(images)
        logits = output["out"]
        return logits