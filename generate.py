import math

import torch, torchvision

from model import StyledGenerator

class RealImageGenerator(object):
    def __init__(self, path='Data/stylegan-256px-new.model', size=256, device='cuda'):
        self.device = device
        self.size = size

        self.generator = StyledGenerator(512).to(device)
        self.generator.load_state_dict(torch.load(path, map_location=device)['g_running'])
        self.generator.eval()

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
    @torch.no_grad()
    def generate(self, codes):
        image, state = self.generator(
            codes,
            step=self.step,
            alpha=1,
            mean_style=self.mean_style,
            style_weight=0.7
        )

        image.clamp_(min=-1, max=1)
        image.add_(1).div_(2 + 1e-5)

        return image, state

    @torch.no_grad()
    def sample(self, n_sample, with_last_states=True):
        codes = torch.randn(n_sample, 512).to(self.device)
        image, state = self.generate(codes)
        
        if with_last_states:
            return image, state
        else:
            return image

# Test Code
if __name__ == '__main__':
    generator = RealImageGenerator(path='Data/stylegan-256px-new.model', size=256, device='cpu')

    img, states = generator.sample(4)
    print('Shape of Last Hidden States: '+ str(states.shape))
    
    # show images
    import matplotlib.pyplot as plt
    import numpy as np

    img = torchvision.utils.make_grid(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
   