"""GR-ConvNet (Kumra et al., IROS 2020), adapted from the MIT-licensed code in
https://github.com/Fsoft-AIC/LGD (itself derived from the original GR-ConvNet).

Changes: a fifth head predicts the jaw size h (the original fixes h = w / 2), and
an optional additive text input reproduces the entangled "GR-ConvNet + CLIP"
baseline of LGD -- without the .detach() on visual features found in that code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ("pos", "cos", "sin", "w", "h")


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c)
        self.conv2, self.bn2 = nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class GRConvNet(nn.Module):
    def __init__(self, in_ch=3, c=32, text_dim=0):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(in_ch, c, 9, 1, 4), nn.BatchNorm2d(c)
        self.conv2, self.bn2 = nn.Conv2d(c, 2 * c, 4, 2, 1), nn.BatchNorm2d(2 * c)
        self.conv3, self.bn3 = nn.Conv2d(2 * c, 4 * c, 4, 2, 1), nn.BatchNorm2d(4 * c)
        self.res = nn.Sequential(*[ResidualBlock(4 * c) for _ in range(5)])
        self.conv4, self.bn4 = nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1, output_padding=1), nn.BatchNorm2d(2 * c)
        self.conv5, self.bn5 = nn.ConvTranspose2d(2 * c, c, 4, 2, 2, output_padding=1), nn.BatchNorm2d(c)
        self.conv6 = nn.ConvTranspose2d(c, c, 9, 1, 4)
        self.heads = nn.ModuleDict({k: nn.Conv2d(c, 1, 2) for k in HEADS})
        self.text = None
        if text_dim:                                   # entangled baseline: broadcast-add text
            self.text = nn.Sequential(nn.Linear(text_dim, 256), nn.GELU(), nn.Linear(256, 4 * c))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, text=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res(x)                                        # (B, 128, 56, 56)
        if self.text is not None:
            x = x + self.text(text)[:, :, None, None]
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)                                      # (B, 32, 225, 225)
        return {k: h(x)[:, 0] for k, h in self.heads.items()}  # each (B, 224, 224)


def g_loss(pred, maps):
    """pos: BCE on logits. Geometry (cos, sin, w, h): smooth-L1 on grasp pixels only.

    GR-ConvNet regresses all five maps on every pixel; with grasps covering only
    ~3-10% of a Grasp-Anything image, the width/height maps then collapse towards
    zero (in a 70-scene overfitting test the predicted opening stayed at ~15 px vs
    ~80 px ground truth). Geometry is only defined where a grasp exists, so we
    supervise it there. maps: (B, 5, H, W).
    """
    pos_t = maps[:, 0].contiguous()
    m = pos_t > 0.5
    parts = {"pos": F.binary_cross_entropy_with_logits(pred["pos"].contiguous(), pos_t)}
    for i, k in enumerate(HEADS[1:], start=1):
        pk = pred[k].contiguous()
        parts[k] = F.smooth_l1_loss(pk[m], maps[:, i].contiguous()[m]) if m.any() else pk.sum() * 0
    return sum(parts.values()), parts
