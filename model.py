import torch
import torch.nn as nn


# Input dim :(Batch, 100*100 ~ U[-1,1])
# Output dim: (Batch, channel:3, Height=64, Width=128)
class Generator(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(Generator, self).__init__()
        self.linear = nn.Linear(z_dim, 8192 * 2)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 3, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = self.linear(inputs)
        output = output.view(-1, 512, 4, 8)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        return output


class Discriminator(nn.Module):
    def __init__(self, sigma: float) -> None:
        super(Discriminator, self).__init__()
        self.sigma = sigma
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        self.linear = nn.Linear(4 * 8 * 512, 1)

    def add_noise(self, inputs: torch.Tensor) -> torch.Tensor:
        """add random noise witha  uniform random sigma multiplier"""
        noise = torch.zeros_like(inputs).normal_(0, torch.rand(1).item() * self.sigma)
        return torch.relu(inputs + noise)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = self.add_noise(inputs)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)

        return output
