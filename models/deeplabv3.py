from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class DeepLabV3MobileNet(nn.Module):
    """
    Wraps the DeepLabV3 model for use in this training suite
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=num_classes
    )


    def forward(self, images):
        """
        Defines the forward pass of the model
        :param images:  The images being shown to the model
        :return:        Logits of shape [batch_size, num_classes, height, width]
        """
        output = self.net(images)
        logits = output["out"]
        return logits
