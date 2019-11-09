import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from datasets import SketchDataLoader
from model import Discriminator, Generator


def mask_image(image):
    mask = torch.ones_like(image)
    mask[:, :, :, int(mask.size(-1) / 2) :] = 0
    return image * mask


def random_uniform(
    r1: int, r2: int, batch: int, dim: int, device: torch.device
) -> torch.Tensor:
    return ((r1 - r2) * torch.rand(batch, dim) + r2).to(device)


def compute_loss(generated_image, sketch, d_model, _lambda=0.5):
    # sketch -> [b, 3, h, 2*w]
    # generated_image -> [b, 3, h, 2*w]

    g_loss = _lambda * torch.log(1.0 - d_model(generated_image).sigmoid()).mean()

    mse_criterion = nn.MSELoss()
    kl_loss = mse_criterion(mask_image(generated_image), mask_image(sketch))

    return g_loss, kl_loss


def main(args):
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    train_loader = SketchDataLoader(
        root="~/Data/Datasets/Flickr-Face-HQ",
        train=not args.debug,
        sketch_type="XDoG",
        size=64,
        size_from="thumbs",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        device=device,
    )
    test_loader = SketchDataLoader(
        root="~/Data/Datasets/Flickr-Face-HQ",
        train=False,
        sketch_type="XDoG",
        size=64,
        size_from="thumbs",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        device=device,
    )

    netG = Generator(args.z_dim).to(device)
    netD = Discriminator().to(device)

    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.g_lr)
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.d_lr)

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
            real_output = netD(real).view(-1)

            ## Train with all-fake batch, Generate batch of latent vectors, then fake batch
            noise = random_uniform(-1, 1, b_size, args.z_dim, device)
            fake = netG(noise)
            fake_output = netD(fake.detach()).view(-1)

            # calculate D's loss on the real and fake batch, D wants to minimize real (positive) values
            # and also minimize fake (negative) values
            errD = -torch.mean(real_output) + torch.mean(fake_output)
            writer.add_scalar("Loss/D", errD.item(), niter)
            errD.backward()
            optimizerD.step()

            # clip weights according to WGAN
            for p in netD.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            if niter % args.g_iter == 0:
                # (2) Update G network: after updating D, perform another forward pass through D
                netG.zero_grad()
                output = netD(fake).view(-1)

                errKL = kl_criterion(mask_image(fake), mask_image(real))

                # the mean output is negative because G wants D to output a real value (positive), but we need to
                # minimize this loss for G
                errG = (args.train_lambda * -torch.mean(output)) + errKL
                writer.add_scalar("Loss/G", errG.item(), niter)
                writer.add_scalar("Loss/kl", errKL.item(), niter)

                # Calculate gradients for G, and update
                errG.backward()
                optimizerG.step()

            niter += 1
            str = f"errD: {errD.item():06.4f} errG: {errG.item():06.4f} errKl: {errKL.item():06.4f}"
            loss_log.set_description_str(str)

            if niter % 500 == 0 or args.debug:
                x = vutils.make_grid(fake, normalize=True, scale_each=True)
                writer.add_image("Image/main", x, niter)

            if niter % 5000 == 0 or args.debug:

                eval_niter += 1
                # initialize N uniform per batch
                # sketch -> [b, 3, h, 2*w]

                z_list = []
                mse_criterion = nn.MSELoss(reduction="none")

                # NOTE: this uses the same sketch throughout time so we can monitor the performance
                # on the same data
                for i in range(fixed_sketch.size(0)):
                    z = random_uniform(-1, 1, args.eval_N, args.z_dim, device)
                    initial_sketch = netG(z)
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

                    fake = netG(z)
                    output = netD(fake).view(-1)
                    errKL = kl_criterion(mask_image(fake), fixed_sketch)
                    errG = (args.eval_lambda * -torch.mean(output)) + errKL
                    errG.backward()
                    optimizer.step()

                fake = netG(z)
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

    # training
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=8, type=int, help="batch_size")
    parser.add_argument("--g_lr", default=2e-4, type=float, help="g lr")
    parser.add_argument("--d_lr", default=2e-4, type=float, help="d lr")
    parser.add_argument("--train_lambda", default=0.01, type=float, help="lamda")
    parser.add_argument(
        "--clip_value", default=0.01, type=int, help="clip min and max values"
    )
    parser.add_argument(
        "--g_iter", default=5, type=int, help="number of D iterations per G iteration"
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

    parser.add_argument("--z_dim", type=int, default=100)

    args = parser.parse_args()

    # set model dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = os.path.abspath(save_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
