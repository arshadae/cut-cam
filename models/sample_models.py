import torch.nn as nn
from torchvision.models import resnet18

class MyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.netA = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.netB = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.netA(x)
        x = self.netB(x)
        return x
    

class MyGenerator2(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18 and remove final layers
        resnet = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool & fc

        # Freeze encoder if needed
        for param in self.encoder.parameters():
            param.requires_grad = False  # Comment out this line to make it trainable

        # Your custom trainable decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32 → 64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64 → 128
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 128 → 256
            nn.Tanh()  # or nn.Sigmoid() or nothing, depending on your output normalization
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out
