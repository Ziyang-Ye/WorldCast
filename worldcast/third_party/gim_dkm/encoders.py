"""DKM's encoder: torchvision's ResNet-50 without its classifier, features at strides 1 to 32."""

import torch.nn as nn
import torchvision.models as tvm


class ResNet50(nn.Module):
    def __init__(self, high_res=False, freeze_bn=True) -> None:
        super().__init__()
        self.net = tvm.resnet50(weights=None, replace_stride_with_dilation=[False, False, False])
        del self.net.fc
        self.high_res = high_res
        self.freeze_bn = freeze_bn

    def forward(self, x):
        net = self.net
        feats = {1: x}
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        feats[2] = x
        x = net.maxpool(x)
        x = net.layer1(x)
        feats[4] = x
        x = net.layer2(x)
        feats[8] = x
        x = net.layer3(x)
        feats[16] = x
        x = net.layer4(x)
        feats[32] = x
        return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
