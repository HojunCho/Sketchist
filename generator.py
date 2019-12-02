import torch
from torch import nn

from model import Discriminator, Generator, RealImageGenerator
from utils import mask_image, random_uniform

z_dim = 512
eval_N = 10
lr = 0.1
momentum = 0.9
eval_iterations = 500
eval_lambda = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

netRealG = RealImageGenerator(device=device)
netG = Generator().to(device)
netG.load_state_dict(torch.load("./Data/g_10_checkpoint.pt", map_location=device))
netD = Discriminator().to(device)
netD.load_state_dict(torch.load("./Data/d_10_checkpoint.pt", map_location=device))

kl_criterion = nn.MSELoss()


def generate(sketch: torch.Tensor) -> torch.Tensor:
    global netG
    global kl_criterion

    # add space for the generated image to the tensor
    sketch = torch.cat((sketch, torch.zeros_like(sketch)), 2).to(device)
    # TODO: are the first three channels correct
    sketch = sketch[:3, :, :]

    z_list = []

    # generate random noise -> generate from noise -> compare sketches
    z = random_uniform(-1, 1, eval_N, z_dim, device)
    initial_sketch, states = netRealG.generate(z)
    initial_sketch = torch.cat([netG(states), initial_sketch], dim=-1)

    initial_sketch_kl_loss = kl_criterion(
        mask_image(initial_sketch), sketch.unsqueeze(0).repeat(eval_N, 1, 1, 1)
    )

    initial_sketch_kl_loss = initial_sketch_kl_loss.mean(-1).mean(-1).mean(-1)

    best_index = initial_sketch_kl_loss.argmax(dim=0)
    z = z[best_index]
    z_list.append(z)

    z = torch.stack(z_list, dim=0)
    z_as_params = [z]
    optimizer = torch.optim.SGD(z_as_params, lr=lr, momentum=momentum)
    for _ in range(eval_iterations):
        netRealG.generator.zero_grad()
        netG.zero_grad()
        netD.zero_grad()
        optimizer.zero_grad()

        fake, states = netRealG.generate(z)
        fake = torch.cat([netG(states), fake], dim=-1)
        output = netD(fake).view(-1)
        errKL = kl_criterion(mask_image(fake), sketch.unsqueeze(0))
        errG = (eval_lambda * -torch.mean(output)) + errKL
        errG.backward()
        optimizer.step()

    return netRealG.generate(z)[0].cpu()


if __name__ == "__main__":
    from datasets import SketchDataLoader
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import numpy as np

    def show_images(imgs: torch.Tensor):
        imgs = make_grid(imgs * 0.5 + 0.5)
        imgs = imgs.cpu().detach().numpy()
        plt.imshow(np.transpose(imgs, (1, 2, 0)))
        plt.show()

    test_loader = SketchDataLoader(
        root="~/Data/Datasets/Flickr-Face-HQ",
        train=False,
        sketch_type="XDoG",
        size=256,
        size_from="images",
        batch_size=2,
        shuffle=True,
        num_workers=2,
        device=device,
    )

    sketch = next(iter(test_loader))[0, :, :, :256]
    show_images(sketch)
    image = generate(sketch)
    show_images(image)
