import argparse
import os
from datetime import datetime
from typing import Tuple

import numpy as np  # type: ignore
import torch
import torch.nn as nn
import torchvision.utils as vutils  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from tensorboardX import SummaryWriter  # type: ignore
# import torch.autograd as autograd
from torch.autograd import Variable, grad
from torch.optim import RMSprop  # type: ignore
from tqdm import tqdm, trange  # type: ignore

from datasets import FFHQ
from datasets import SketchDataLoader
from model import Discriminator, Generator, RealImageGenerator
from utils import mask_image, random_uniform


def get_gradient_penalty(
    D: nn.Module, real: torch.Tensor, fake: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, ...]:
    batch_size = real.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real).to(device)
    interpolated = alpha * real.data + (1 - alpha) * fake.data  # type: ignore
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty, and norm. print norm to make sure it stays small
    return ((gradients_norm - 1) ** 2).mean(), gradients.norm(2, dim=1).mean().data


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = SketchDataLoader(
        root="~/Data/Datasets/Flickr-Face-HQ",
        train=not args.debug,
        sketch_type="XDoG",
        size=256,
        size_from="thumbs",
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
        device=device,
    )
    test_loader = SketchDataLoader(
        root="~/Data/Datasets/Flickr-Face-HQ",
        train=False,
        sketch_type="XDoG",
        size=256,
        size_from="thumbs",
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=2,
        device=device,
    )

    realImageG = RealImageGenerator(
        path=args.real_image_generator_path, size=256, device=device
    )
    netG = Generator().to(device)
    # netG.load_state_dict(torch.load("./.dummy/keep/g_10_ninth.pt"))
    netD = Discriminator().to(device)
    # netD.load_state_dict(torch.load("./.dummy/keep/d_10_ninth.pt"))

    optimizerG = torch.optim.Adam(
        netG.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2)
    )
    optimizerD = torch.optim.Adam(
        netD.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2)
    )

    log_dir = os.path.join(
        os.path.join(args.save_dir, f"{datetime.now().strftime('%Y-%m-%d-%H:%M')}"),
        "tensorboard",
    )
    writer = SummaryWriter(log_dir=log_dir)

    dataiter = iter(test_loader)
    fixed_sketch = next(dataiter)

    # write an example image before masking
    img = vutils.make_grid(fixed_sketch, normalize=True, scale_each=True)
    writer.add_image("Image/example", img, 0)

    fixed_sketch = mask_image(fixed_sketch).to(device)

    # write an example image after masking the image
    img = vutils.make_grid(fixed_sketch, normalize=True, scale_each=True)
    writer.add_image("Image/fixed_sketch", img, 0)

    loss_log = tqdm(total=0, bar_format="{desc}", position=2)

    kl_criterion = nn.MSELoss()

    niter = 0
    eval_niter = 0

    for epoch in trange(int(args.epochs), desc="Epoch", position=0):
        for real in tqdm(train_loader, desc="Train iter", leave=False, position=1):
            ## Train D with all-real batch
            netD.zero_grad()
            # Format batch
            real = real.to(device)
            b_size = real.size(0)

            # Forward pass real batch through D
            real_output = torch.mean(netD(real).view(-1))

            ## Train with all-fake batch, Generate batch of latent vectors, then fake batch
            noise = random_uniform(-1, 1, b_size, args.z_dim, device)
            with torch.no_grad():
                fake, hidden = realImageG.generate(noise)
            fake_sketch = netG(hidden)
            fake = torch.cat([fake_sketch, fake], dim=-1)
            fake_output = torch.mean(netD(fake.detach()).view(-1))

            gradient_penalty, norm = get_gradient_penalty(
                netD, real.data, fake.data, device
            )
            # calculate D's loss on the real and fake batch, D wants to minimize real (positive) values
            # and also minimize fake (negative) values
            errD = -real_output + fake_output + args.lambda_gp * gradient_penalty
            writer.add_scalar("Loss/D", errD.item(), niter)
            errD.backward()
            optimizerD.step()

            if niter % args.g_iter == 0:
                # (2) Update G network: after updating D, perform another forward pass through D
                netG.zero_grad()
                output = netD(fake).view(-1)

                # the mean output is negative because G wants D to output a real value (positive), but we need to
                # minimize this loss for G
                errG = -torch.mean(output)
                writer.add_scalar("Loss/G", errG.item(), niter)

                # Calculate gradients for G, and update
                errG.backward()
                optimizerG.step()

            niter += 1
            str = f"errD: {errD.item():06.4f} errG: {errG.item():06.4f}, gradient norm: {norm.item():.2f}"
            loss_log.set_description_str(str)

            if niter % 500 == 0 or args.debug:
                x = vutils.make_grid(fake, normalize=True, scale_each=True)
                writer.add_image("Image/main", x, niter)

            if niter % 5000 == 0 or args.debug:

                eval_niter += 1
                # initialize N uniform per batch sketch -> [b, 3, h, 2*w]
                z_list = []
                mse_criterion = nn.MSELoss(reduction="none")

                # NOTE: this uses the same sketch throughout time so we can monitor the performance
                # on the same data
                for i in range(fixed_sketch.size(0)):
                    z = random_uniform(-1, 1, args.eval_N, args.z_dim, device)
                    _, hidden = realImageG.generate(z)
                    initial_sketch = netG(hidden)
                    initial_sketch = torch.cat(
                        [initial_sketch, torch.zeros_like(initial_sketch)], dim=-1
                    )
                    initial_sketch_kl_loss = mse_criterion(
                        mask_image(initial_sketch),
                        fixed_sketch[i].unsqueeze(0).repeat(args.eval_N, 1, 1, 1),
                    )
                    initial_sketch_kl_loss = (
                        initial_sketch_kl_loss.mean(-1).mean(-1).mean(-1)
                    )
                    best_index = initial_sketch_kl_loss.argmax(dim=0)
                    z = z[best_index]
                    z_list.append(z)

                z = torch.stack(z_list, dim=0)
                z_as_params = [z]
                optimizer = torch.optim.SGD(
                    z_as_params, lr=args.eval_lr, momentum=args.eval_momentum
                )
                for _ in range(args.eval_iterations):
                    netG.zero_grad()
                    netD.zero_grad()
                    optimizer.zero_grad()

                    fake, hidden = realImageG.generate(z)
                    fake_sketch = netG(hidden)
                    fake = torch.cat([fake_sketch, fake], dim=-1)
                    output = netD(fake).view(-1)
                    errKL = kl_criterion(mask_image(fake), fixed_sketch)
                    errG = (args.eval_lambda * -torch.mean(output)) + errKL
                    errG.backward()
                    optimizer.step()

                fake, hidden = realImageG.generate(z)
                fake_sketch = netG(hidden)
                fake = torch.cat([fake_sketch, fake], dim=-1)
                x = vutils.make_grid(fake, normalize=True, scale_each=True)
                writer.add_image("Image/eval", x, eval_niter)

        if epoch % 10 == 0 or args.debug:
            g_checkpoint_dir = os.path.join(
                args.save_dir, "g_{}_checkpoint.pt".format(epoch)
            )
            d_checkpoint_dir = os.path.join(
                args.save_dir, "d_{}_checkpoint.pt".format(epoch)
            )
            torch.save(netG.state_dict(), g_checkpoint_dir)
            torch.save(netD.state_dict(), d_checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)

    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--save_dir", default=".dummy", type=str)
    parser.add_argument("--real_image_generator_path", default="./stylegan-256px-new.model", type=str)

    # training
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--train_batch_size", default=12, type=int, help="batch_size")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="batch_size")
    parser.add_argument("--g_lr", default=2e-4, type=float, help="g lr")
    parser.add_argument("--d_lr", default=2e-4, type=float, help="d lr")
    parser.add_argument("--train_lambda", default=0.01, type=float, help="lamda")
    parser.add_argument(
        "--clip_value", default=0.01, type=int, help="clip min and max values"
    )
    parser.add_argument(
        "--g_iter", default=5, type=int, help="number of D iterations per G iteration"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta 1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta 2 for Adam optimizer"
    )

    # evaluation
    parser.add_argument("--eval_N", default=10, type=int, help="N")
    parser.add_argument("--eval_lambda", default=0.01, type=float, help="lamda")
    parser.add_argument(
        "--eval_iterations", default=500, type=int, help="eval iteration"
    )
    parser.add_argument("--eval_lr", default=0.1, type=float, help="eval iteration")
    parser.add_argument(
        "--eval_momentum", default=0.9, type=float, help="eval iteration"
    )

    parser.add_argument("--z_dim", type=int, default=512)
    parser.add_argument(
        "--lambda_gp",
        type=int,
        default=10,
        help="lambda coefficient for gradient penalty",
    )

    args = parser.parse_args()

    # set model dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = os.path.abspath(save_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore

    main(args)
