import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from model import Generator, Discriminator
from datasets import SketchDataLoader


def Embedding(sketch,emb_dim):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    Emb=nn.Linear(3*64*64,emb_dim).to(device)
    return Emb(sketch)

class Generator(nn.Module):

    def __init__(self, z_dim,emb_dim):
        super(Generator, self).__init__()
        def make_sequential(in_channels, out_channels, kernel=5, stride=2, padding=2,output_padding=1):
            return nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel,stride=stride, padding=padding,output_padding=output_padding),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU())
        self.linear=nn.Linear(int(z_dim+emb_dim), 8192)
        self.layer1=make_sequential(512,256)
        self.layer2=make_sequential(256,128)
        self.layer3=make_sequential(128,64)
        self.layer4=nn.Sequential(
            nn.ConvTranspose2d(64,3,kernel_size=5,stride=2, padding=2,output_padding=1),
            nn.Tanh()
        )
    #### sketch is alredy embedded in Embedding fuction
    def forward(self,inputs,sketch):
        inputs=torch.cat((inputs,sketch),-1)
        #### concat sketch and latent variable 
        output=self.linear(inputs)
        output=output.view(-1,512,4,4)
        output=self.layer1(output)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)

        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def make_sequential(in_channels, out_channels, kernel=5, stride=2, padding=2):
            return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=kernel,stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU())

        self.layer1=make_sequential(3,64)
        self.layer2=make_sequential(64,128)
        self.layer3=make_sequential(128,256)
        self.layer4=make_sequential(256,512)
        self.linear=nn.Linear(4*8*512,1)


    def forward(self, inputs):
        #### input size: batch, 3, 64,128 : find conditional probability by joint image
        output=self.layer1(inputs)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=output.view(output.shape[0],-1)
        output=self.linear(output).sigmoid()

        return output
        
def main(args):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    train_loader = SketchDataLoader(root='~/Data/Datasets/Flickr-Face-HQ', train=not args.debug, sketch_type='XDoG', size=64, size_from='thumbs',
                                    batch_size=args.batch_size, shuffle=True, num_workers=2, device=device)
    test_loader = SketchDataLoader(root='~/Data/Datasets/Flickr-Face-HQ', train=False, sketch_type='XDoG', size=64, size_from='thumbs',
                                    batch_size=args.batch_size, shuffle=True, num_workers=2, device=device)
    

    dataiter = iter(test_loader)
    fixed_sketch = next(dataiter)
    fixed_sketch=fixed_sketch.to(device)

    netG = Generator(args.z_dim, args.emb_dim).to(device)
    netD = Discriminator().to(device)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.g_lr, betas=(args.g_beta, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.d_lr, betas=(args.d_beta, 0.999))

    log_dir = os.path.join(args.save_dir, "tensorboard_cgan")
    writer = SummaryWriter(log_dir=log_dir)

    loss_log = tqdm(total=0, bar_format='{desc}', position=2)
    #### MSE vs Binary Cross Entorpy?
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
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.z_dim, device=device)
            # Generate fake image batch with G
            emb_sketch=real[:,:,:,:real.size(-1)//2]
            emb_sketch=emb_sketch.reshape(-1,3*64*64)
            # emb_sketch=emb_sketch.type(torch.FloatTensor)
            # print(emb_sketch.shape)
            emb_sketch=Embedding(emb_sketch,args.emb_dim)
            fake = netG(noise,emb_sketch)
            label.fill_(fake_label)
            # Classify all fake batch with D
            D_input=torch.cat((real[:,:,:,:real.size(-1)//2],fake),-1)
            output = netD(D_input.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(D_input).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            
            # Update G
            optimizerG.step()

            niter += 1
            writer.add_scalars('data/loss_group',
                               {'D_x': D_x,
                                'D_G_z1': D_G_z1,
                                'D_G_z2': D_G_z2,
                                },
                               niter)
            str = 'D_x : {:06.4f} D_G_z1 : {:06.4f} D_G_z2 : {:06.4f}'

            str = str.format(float(D_x), float(D_G_z1),
                             float(D_G_z2))
            loss_log.set_description_str(str)

            ####It needs debuging
            if niter % 500 ==0 or args.debug:
                eval_niter+=1
                target_sketch=real[0:args.eval_N,:,:,:real.size(-1)].unsqueeze(0) #(1,3,64,64)
                target_sketch=target_sketch.view(-1,3*64*64)
                emb_target=Embedding(target_sketch,args.emb_dim)
                z = torch.rand(args.eval_N, args.z_dim).to(device)
                fake = netG(z,emb_target[:args.eval_N])
                x = vutils.make_grid(fake, normalize=True, scale_each=True)
                writer.add_image('Image', x, eval_niter)
        
        if epoch % 40 == 0 or args.debug:
            g_checkpoint_dir = os.path.join(args.save_dir, "g_cgan_{}_checkpoint.pt".format(epoch))
            d_checkpoint_dir = os.path.join(args.save_dir, "d_cgan_{}_checkpoint.pt".format(epoch))
            torch.save(netG.state_dict(), g_checkpoint_dir)
            torch.save(netD.state_dict(), d_checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)

    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument("--save_dir", default=".dummy", type=str)

    #training
    parser.add_argument("--epochs", default=320, type=int)
    parser.add_argument("--batch_size", default=256, type=int, help="batch_size")
    parser.add_argument("--g_iter", default=1, type=int, help="number of genrator iter")
    parser.add_argument("--d_iter", default=4, type=int, help="number of discriminator iter")
    parser.add_argument("--g_lr", default=2e-4, type=float, help="g lr")
    parser.add_argument("--d_lr", default=4e-6, type=float, help="d lr")
    parser.add_argument("--g_beta", default=0.5, type=float, help="g beta")
    parser.add_argument("--d_beta", default=0.5, type=float, help="d beta")
    #evaluation
    parser.add_argument("--eval_N", default=10, type=int, help="N")
    parser.add_argument("--eval_iterations", default=500, type=int, help="eval iteration")
    parser.add_argument("--eval_lr", default=0.1, type=float, help="eval iteration")
    parser.add_argument("--eval_momentum", default=0.9, type=float, help="eval iteration")


    parser.add_argument('--z_dim', type=int, default=100*100)
    parser.add_argument('--emb_dim', type=int, default=1000)

    args = parser.parse_args()

    # set model dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = os.path.abspath(save_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)



