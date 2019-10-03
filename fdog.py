import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import math

from datasets import FFHQ

class Sobel(nn.Module):

    def __init__(self):
        super(Sobel, self).__init__()        
        kernel = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ])
        self.kernel = torch.stack([kernel, kernel.transpose(0, 1)])

    def forward(self, inputs):
        kernel = self.kernel.to(device=inputs.device, dtype=inputs.dtype)
        kernel = kernel.unsqueeze(dim=1)

        return F.conv2d(inputs, kernel, padding=1)


class ETF(nn.Module):

    def __init__(self, mu, iterations):
        super(ETF, self).__init__()
        self.mu = mu
        self.iterations = iterations

        self.sobel = Sobel()
        self.padding = nn.ZeroPad2d(mu)

    def forward(self, inputs):
        sobel = self.sobel(inputs)
        sobel_mag = torch.norm(sobel, dim=1, keepdim=True)
        sobel_mag /= torch.max(sobel_mag)
        sobel_mag_padded = self.padding(sobel_mag)
        sobel_X = sobel_mag

        tang = sobel.flip(dims=[1]) * torch.tensor([-1., 1.]).view(1, 2, 1, 1)
        tang_mag = torch.norm(tang, dim=1, keepdim=True)
        tang_mag[tang_mag == 0] = 1
        tang /= tang_mag

        for iteration in range(self.iterations):
            for ori in ['Vertical', 'Horizontal']:
                tang_padded = self.padding(tang)
                tang_X = tang
                tang = torch.zeros_like(tang)
                total_weight = torch.zeros_like(tang)

                for i in range(2 * self.mu + 1):
                    if ori == 'Vertical':
                        tang_Y = tang_padded[:, :, i:i-2*self.mu or None, self.mu:-self.mu]
                        sobel_Y = sobel_mag_padded[:, :, i:i-2*self.mu or None, self.mu:-self.mu]
                    else:
                        tang_Y = tang_padded[:, :, self.mu:-self.mu, i:i-2*self.mu or None]
                        sobel_Y = sobel_mag_padded[:, :, self.mu:-self.mu, i:i-2*self.mu or None]

                    tang = tang_Y * (sobel_Y - sobel_X + 1) * (tang_X * tang_Y).sum(dim=2, keepdim=True) / 2.
                    tang_mag = torch.norm(tang, dim=1, keepdim=True)
                    tang_mag[tang_mag == 0] = 1
                    tang /= tang_mag

        return tang


def _guassian_pdf(x, sigma):
    return math.exp(- x ** 2 / (sigma ** 2 * 2)) / (math.sqrt(math.pi * 2) * sigma)


class DoG(nn.Module):

    def __init__(self, sigma_c, rho):
        super(DoG, self).__init__()
        self.sigma_c = sigma_c
        self.sigma_s = sigma_c * 1.6
        self.rho = rho

        self.max_T = math.floor(sigma_c * 3)
        self.delta = 1

    def forward(self, images, etf):
        b, c, x, y = images.shape
        indices = torch.stack(torch.meshgrid(torch.arange(0, x, dtype=images.dtype), torch.arange(0, y, dtype=images.dtype))).repeat(b, 1, 1, 1)
        
        dog = torch.zeros_like(images)
        per = etf.flip(dims=[1]) * torch.tensor([-1., 1.]).view(1, 2, 1, 1)
        total_weight = 0.
        for t in range(-self.max_T, self.max_T + 1):
            points = indices + self.delta * per * t
            points[:,0,:,:] = points[:,0,:,:].clamp(min=0, max=x-1)
            points[:,1,:,:] = points[:,1,:,:].clamp(min=0, max=y-1)
            points = torch.round(points).to(dtype=torch.long)

            il_s = torch.gather(images.view(b, c, x * y), 2, (points[:,0,:,:] * y + points[:,1,:,:]).view(b, c, x * y)).view(b, c, x, y)
            gauss_weight = _guassian_pdf(t, self.sigma_c) - self.rho * _guassian_pdf(t, self.sigma_s)
            total_weight += gauss_weight
            dog += il_s * gauss_weight

        dog /= total_weight
        return dog
            

class FDoG(nn.Module):

    def __init__(self, mu=10, iterations=3, sigma_c=3.0, sigma_m=10.0, rho=0.99, tau=0.7):
        super(FDoG, self).__init__()
        self.etf = ETF(mu, iterations)
        self.dog = DoG(sigma_c, rho)
        self.sigma_m = sigma_m
        self.tau = tau

        self.max_S = math.floor(sigma_m * 3)
        self.delta = 1
    
    def forward(self, images):
        etf = self.etf(images)
        dog = self.dog(images, etf=etf)

        b, c, x, y = images.shape

        fdog = torch.zeros_like(images)
        total_weight = 0.
        for s_dir in [-1, 1]:
            indices = torch.stack(torch.meshgrid(torch.arange(0, x), torch.arange(0, y))).repeat(b, 1, 1, 1)
            points = indices.to(dtype=images.dtype)
            for s in range(0 if s_dir == 1 else 1, self.max_S + 1):
                if s != 0:
                    p_etf = torch.stack([torch.gather(etf[:,[i],:,:].view(b, x * y), 1, (indices[:,0,:,:] * y + indices[:,1,:,:]).view(b, x * y)).view(b, x, y) for i in range(2)], dim=1)
                    points += self.delta * p_etf * s_dir
                    points[:,0,:,:] = points[:,0,:,:].clamp(min=0, max=x-1)
                    points[:,1,:,:] = points[:,1,:,:].clamp(min=0, max=y-1)
                    indices = torch.round(points).to(dtype=torch.long)
            
                f_s = torch.gather(dog.view(b, c, x * y), 2, (indices[:,0,:,:] * y + indices[:,1,:,:]).view(b, c, x * y)).view(b, c, x, y)
                gauss_weight = _guassian_pdf(s, self.sigma_m)
                total_weight += gauss_weight
                fdog += f_s * gauss_weight

        fdog /= total_weight
        fdog = ~((fdog < 0) * (1 + torch.tanh(fdog) < self.tau))
        return fdog.to(dtype=torch.long)


#---------------------------------------------------------------------
# Test code
def _show_images(images):
    npimg = images.numpy()
    plt.imshow(np.squeeze(npimg), cmap='gray')
    plt.show()

def _show_arrows(vector_map):
    if vector_map.shape[2] > 500:
        return
    np_map = vector_map.numpy()

    c, y, x = np_map.shape
    Q = plt.quiver(np.arange(0, x), np.arange(0, y), np_map[1,:,:], -1 * np_map[0,:,:], scale=100)
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    trainset = FFHQ(root='~/Data/Flickr-Face-HQ', train=False, size='images', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)

    dataiter = iter(trainloader)
    images = dataiter.next().to(device)
    _show_images(images[0,:,:,:])

    fdog = FDoG()
    fdog.to(device)
    out = fdog(images)
    _show_images(out[0,:,:,:])

    torchvision.utils.save_image(images[0,:,:,:] / 2 + 0.5, 'test_in.png')
    torchvision.utils.save_image(out[0,:,:,:], 'test_out.png')

#---------------------------------------------------------------------