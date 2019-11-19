import torch
from torch import nn

from model import Discriminator, Generator
from utils import mask_image, random_uniform

z_dim = 10000
eval_N = 10
lr = 0.1
momentum = 0.9
eval_iterations = 50
eval_lambda = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator(z_dim).to(device)
netG.load_state_dict(torch.load("./.dummy/g_270_checkpoint.pt"))
netD = Discriminator().to(device)
netD.load_state_dict(torch.load("./.dummy/d_270_checkpoint.pt"))

kl_criterion = nn.MSELoss()


def generate(sketch: torch.Tensor) -> torch.Tensor:
    global netG
    global kl_criterion

    # add space for the generated image to the tensor
    sketch = torch.cat((sketch, torch.zeros_like(sketch)), 2).to(device)

    z_list = []

    # generate random noise -> generate from noise -> compare sketches
    z = random_uniform(-1, 1, eval_N, z_dim, device)
    initial_sketch = netG(z)

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
        netG.zero_grad()
        netD.zero_grad()
        optimizer.zero_grad()

        fake = netG(z)
        output = netD(fake).view(-1)
        errKL = kl_criterion(mask_image(fake), sketch.unsqueeze(0))
        errG = (eval_lambda * -torch.mean(output)) + errKL
        errG.backward()
        optimizer.step()

    return netG(z).cpu()
