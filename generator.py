import torch
from torch import nn

from model import (Discriminator, Generator, RealImageDiscriminator,
                   RealImageGenerator)
from utils import mask_image, random_uniform

z_dim = 512
eval_N = 20
lr = 0.1
momentum = 0.9
eval_iterations = 80
eval_lambda = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

netRealG = RealImageGenerator(device=device)
netG = Generator().to(device)
netG.load_state_dict(torch.load("./Data/g_20_checkpoint.pt", map_location=device))

netRealD = RealImageDiscriminator(device=device)
netD = Discriminator().to(device)
netD.load_state_dict(torch.load("./Data/d_20_checkpoint.pt", map_location=device))

kl_criterion = nn.MSELoss()


def generate(sketch: torch.Tensor, plot=False, return_video=False) -> torch.Tensor:
    if plot:
        from torchvision.utils import make_grid
        import matplotlib.pyplot as plt
        import numpy as np

        def show_images(imgs: torch.Tensor):
            imgs = make_grid(imgs * 0.5 + 0.5)
            imgs = imgs.cpu().detach().numpy()
            plt.imshow(np.transpose(imgs, (1, 2, 0)))
            plt.show()

    global netG
    global kl_criterion

    # add space for the generated image to the tensor
    sketch = torch.cat((sketch, torch.zeros_like(sketch)), 2).to(device)

    z_list = []

    # generate random noise -> generate from noise -> compare sketches
    z = random_uniform(-1, 1, eval_N, z_dim, device)
    initial_sketch, states = netRealG.generate(z)
    initial_sketch = torch.cat([netG(states), initial_sketch], dim=-1)

    if plot:
        show_images(initial_sketch)

    initial_sketch_kl_loss = kl_criterion(
        mask_image(initial_sketch), sketch.unsqueeze(0).repeat(eval_N, 1, 1, 1)
    )

    initial_sketch_kl_loss = initial_sketch_kl_loss.mean(-1).mean(-1).mean(-1)

    best_index = initial_sketch_kl_loss.argmax(dim=0)
    z = z[best_index]
    z_list.append(z)

    if plot:
        show_images(torch.stack([sketch, initial_sketch[best_index]], dim=0))
    if return_video:
        video = []

    z = torch.stack(z_list, dim=0).requires_grad_(True)
    z_as_params = [z]
    optimizer = torch.optim.SGD(z_as_params, lr=lr, momentum=momentum)
    for i in range(eval_iterations):
        netRealG.generator.zero_grad()
        netRealD.discriminator.zero_grad()
        netG.zero_grad()
        netD.zero_grad()
        optimizer.zero_grad()

        fake, states = netRealG.generate(z)
        output = torch.sigmoid(netRealD.discriminate(fake)).view(-1)
        fake = torch.cat([netG(states), fake], dim=-1)

        # output += netD(fake).view(-1)
        errKL = kl_criterion(fake[:, :, :, :256], sketch.unsqueeze(0)[:, :, :, :256])
        errG = (eval_lambda * -torch.mean(output)) + errKL
        errG.backward()
        optimizer.step()

        if plot and i % 100 == 0:
            show_images(torch.cat([sketch.unsqueeze(0), fake], dim=0))

        if return_video:
            video.append(make_grid(torch.cat([sketch.unsqueeze(0), fake], dim=0) * 0.5 + 0.5))

    fake, states = netRealG.generate(z)
    if plot:
        fake_sketch = torch.cat([netG(states), fake], dim=-1)
        show_images(torch.cat([sketch.unsqueeze(0), fake_sketch], dim=0))
    
    if return_video:
        return fake.cpu(), torch.stack(video).cpu()
    else:
        return fake.cpu()


if __name__ == "__main__":
    from datasets import SketchDataLoader
    from torchvision.utils import make_grid
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    '''
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
    '''
    sketch = plt.imread("./samples/sketch3.png")[:,:,:3]
    sketch = (torch.from_numpy(np.transpose(sketch, axes=[2, 0, 1])) - 0.5) / 0.5

    image, video = generate(sketch, plot=True, return_video=True)
    frames = []
    fig = plt.figure()
    for imgs in video:
        imgs = imgs.cpu().detach().numpy()
        frames.append([plt.imshow(np.transpose(imgs, (1, 2, 0)), animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
    ani.save('./Data/iteration.gif', writer='imagemagick', fps=20)
    plt.show()

