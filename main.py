import argparse
import os

import torch
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from datasets import SketchDataLoader
from model import Discriminator, Generator


def mask_image(image):
    mask = torch.ones_like(image)
    mask[:, :, :, : int(mask.size(-1) / 2)] = 0
    return image * mask


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

    dataiter = iter(test_loader)
    fixed_sketch = next(dataiter)
    fixed_sketch = mask_image(fixed_sketch).to(device)

    netG = Generator(args.z_dim).to(device)
    netD = Discriminator().to(device)

    optimizerG = torch.optim.Adam(
        netG.parameters(), lr=args.g_lr, betas=(args.g_beta, 0.999)
    )
    optimizerD = torch.optim.Adam(
        netD.parameters(), lr=args.d_lr, betas=(args.d_beta, 0.999)
    )

    log_dir = os.path.join(args.save_dir, "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)

    loss_log = tqdm(total=0, bar_format="{desc}", position=2)

    criterion = nn.BCELoss()
    kl_criterion = nn.MSELoss()

    real_label = 1
    fake_label = 0

    niter = 0
    eval_niter = 0
    for epoch in trange(int(args.epochs), desc="Epoch", position=0):
        for real in tqdm(train_loader, desc="Train iter", leave=False, position=1):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            ## Train with all-real batch

            netD.zero_grad()
            # Format batch
            real = real.to(device)
            b_size = real.size(0)
            labels = torch.full((b_size,), real_label, device=device)
            fake_labels = torch.full((b_size,), fake_label, device=device)

            # Forward pass real batch through D
            real_output = netD(real).view(-1)
            D_x = real_output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.z_dim, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            # Classify all fake batch with D
            fake_output = netD(fake.detach()).view(-1)
            D_G_z1 = fake_output.mean().item()

            # calculate D's loss on the real and fake batch
            errD = (
                criterion(real_output, labels) + criterion(fake_output, fake_labels)
            ) / 2
            errD.backward()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, labels)
            # Calculate G's kl loss
            errKL = kl_criterion(mask_image(fake), mask_image(real))
            errG = args.train_lambda * errG + errKL
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            G_kl = errKL.item()
            # Update G
            optimizerG.step()

            niter += 1
            writer.add_scalars(
                "data/loss_group",
                {"D_x": D_x, "D_G_z1": D_G_z1, "D_G_z2": D_G_z2, "G_kl": G_kl},
                niter,
            )
            str = "D_x : {:06.4f} D_G_z1 : {:06.4f} D_G_z2 : {:06.4f} G_kl : {:06.4f}"

            str = str.format(float(D_x), float(D_G_z1), float(D_G_z2), float(G_kl))
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

                for i in range(fixed_sketch.size(0)):
                    z = torch.rand(args.eval_N, args.z_dim).to(device)
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
                label = torch.full((b_size,), real_label, device=device)
                label.fill_(real_label)

                for _ in range(args.eval_iterations):
                    netG.zero_grad()
                    netD.zero_grad()
                    optimizer.zero_grad()

                    fake = netG(z)
                    output = netD(fake).view(-1)
                    errG = criterion(output, label)
                    errKL = kl_criterion(mask_image(fake), fixed_sketch)
                    errG = args.eval_lambda * errG + errKL
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
    parser.add_argument("--g_iter", default=1, type=int, help="number of genrator iter")
    parser.add_argument(
        "--d_iter", default=3, type=int, help="number of discriminator iter"
    )
    parser.add_argument("--g_lr", default=2e-4, type=float, help="g lr")
    parser.add_argument("--d_lr", default=2e-4, type=float, help="d lr")
    parser.add_argument("--g_beta", default=0.5, type=float, help="g beta")
    parser.add_argument("--d_beta", default=0.5, type=float, help="d beta")
    parser.add_argument("--train_lambda", default=0.01, type=float, help="lamda")
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

    parser.add_argument("--z_dim", type=int, default=100 * 100)

    args = parser.parse_args()

    # set model dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = os.path.abspath(save_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
