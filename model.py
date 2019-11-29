import math
import torch, torchvision
import torch.nn as nn
from StyleBasedGAN import StyledGenerator
from StyleBasedGAN import Discriminator as StyledDiscriminator

class RealImageGenerator(object):
    def __init__(self, path='Data/stylegan-256px-new.model', size=256, device='cuda'):
        self.device = device
        self.size = size

        self.generator = StyledGenerator(512).to(device)
        self.generator.load_state_dict(torch.load(path, map_location=device)['g_running'])
        self.generator.train()

        self.mean_style = self.get_mean_style()

    @property
    def step(self):
        return int(math.log(self.size, 2)) - 2

    @torch.no_grad()
    def get_mean_style(self):
        mean_style = None

        for i in range(10):
            style = self.generator.mean_style(torch.randn(1024, 512).to(self.device))

            if mean_style is None:
                mean_style = style

            else:
                mean_style += style

        mean_style /= 10
        return mean_style

    ## codes.shape = (n_batch, 512)
    def generate(self, codes):
        image, state = self.generator(
            codes,
            step=self.step,
            mean_style=self.mean_style,
            style_weight=0.7
        )

        image.clamp_(min=-1, max=1)

        return image, state

    def sample(self, n_sample, with_last_states=True):
        codes = torch.randn(n_sample, 512).to(self.device)
        image, state = self.generate(codes)

        if with_last_states:
            return image, state
        else:
            return image

class RealImageDiscriminator(object):
    def __init__(self, path='Data/stylegan-256px-new.model', size=256, device='cuda'):
        self.device = device
        self.size = size

        self.discriminator = StyledDiscriminator(from_rgb_activate=True).to(device)
        self.discriminator.load_state_dict(torch.load(path, map_location=device)['discriminator'])
        self.discriminator.train()

    @property
    def step(self):
        return int(math.log(self.size, 2)) - 2

    def discriminate(self, images):
        out = self.discriminator(
            images,
            step=self.step
        )

        return out

# Input dim :(Batch, 100*100 ~ U[-1,1])
# Output dim: (Batch, channel:3, Height=64, Width=128)
class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                64, 32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                32, 16, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                16, 8, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                8, 3, kernel_size=5, stride=1, padding=2
            ),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = self.layer1(inputs)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        return output


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([64, 128, 256]),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([128, 64, 128]),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([256, 32, 64]),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([512, 16, 32]),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([1024, 8, 16]),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([2048, 4, 8]),
            nn.LeakyReLU(),
        )

        self.linear = nn.Linear(4 * 8 * 2048, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = self.layer1(inputs)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)

        return output
