import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import math


class Sobel(nn.Module):

    def __init__(self):
        super(Sobel, self).__init__()        
        kernel = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.],
        ])
        self.kernel = torch.stack([kernel, kernel.transpose(0, 1)])

    def forward(self, inputs):
        kernel = self.kernel.to(device=inputs.device, dtype=inputs.dtype)
        kernel = kernel.unsqueeze(dim=1)

        return F.conv2d(inputs, kernel, padding=1)


class ETF(nn.Module):

    def __init__(self, mu, iterations=3, show_progress=False):
        super(ETF, self).__init__()
        self.mu = mu
        self.iterations = iterations

        self.sobel = Sobel()
        self.padding = nn.ZeroPad2d(mu)

        self.__show_progress = show_progress

    @property
    def show_progress(self):
        return self.__show_progress

    def forward(self, inputs):
        sobel = self.sobel(inputs)
        sobel_mag = torch.norm(sobel, dim=1, keepdim=True)
        sobel_mag /= torch.max(sobel_mag)
        sobel_mag_padded = self.padding(sobel_mag)
        sobel_X = sobel_mag

        if self.show_progress:
            _show_arrows(sobel[0,:,:,:])

        tang = sobel.flip(dims=[1]) * torch.tensor([-1., 1.]).view(1, 2, 1, 1)
        tang_mag = torch.norm(tang, dim=1, keepdim=True)
        tang_mag[tang_mag == 0] = 1
        tang /= tang_mag

        if self.show_progress:
            _show_arrows(tang[0,:,:,:])

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

                    tang += tang_Y * (torch.tanh(sobel_Y - sobel_X) + 1) * (tang_X * tang_Y).sum(dim=1, keepdim=True) / 2.

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
        self.rho = rho

        self.delta = 1

    @property
    def sigma_s(self):
        return self.sigma_c * 1.6

    @property
    def max_T(self):
        return math.ceil(self.sigma_s * 3)

    def forward(self, images, etf):
        b, c, y, x = images.shape
        indices = torch.stack(torch.meshgrid(torch.arange(0, y, dtype=images.dtype), torch.arange(0, x, dtype=images.dtype))).repeat(b, 1, 1, 1)
        
        dog = torch.zeros_like(images)
        per = etf.flip(dims=[1]) * torch.tensor([-1., 1.]).view(1, 2, 1, 1)
        total_weight = 0.
        for t in range(-self.max_T, self.max_T + 1):
            points = indices + self.delta * per * t
            points[:,0,:,:] = points[:,0,:,:].clamp(min=0, max=y-1)
            points[:,1,:,:] = points[:,1,:,:].clamp(min=0, max=x-1)
            points = torch.round(points).to(dtype=torch.long)

            il_s = torch.gather(images.view(b, c, y * x), 2, (points[:,0,:,:] * x + points[:,1,:,:]).view(b, c, y * x)).view(b, c, y, x)
            gauss_weight = _guassian_pdf(t, self.sigma_c) - self.rho * _guassian_pdf(t, self.sigma_s)
            total_weight += gauss_weight
            dog += il_s * gauss_weight

        dog /= total_weight
        return dog
            

class FDoG(nn.Module):

    def __init__(self, mu=15, etf_iters=3, fdog_iters=3, sigma_c=2.0, sigma_m=5.0, rho=0.99, tau=0.7, show_progress=False):
        super(FDoG, self).__init__()
        self.etf = ETF(mu, etf_iters, show_progress)
        self.dog = DoG(sigma_c, rho)

        self.sigma_m = sigma_m
        self.tau = tau
        self.fdog_iters = fdog_iters

        self.delta = 1

        self.__show_progress = show_progress

        self.eval()

    @property
    def mu(self):
        return self.etf.mu
    
    @mu.setter
    def mu(self, mu):
        self.etf.mu = mu

    @property
    def etf_iters(self):
        return self.etf.iterations

    @etf_iters.setter
    def etf_iters(self, etf_iters):
        self.etf.iterations = etf_iters
    
    @property
    def sigma_c(self):
        return self.dog.sigma_c

    @sigma_c.setter
    def sigma_c(self, sigma_c):
        self.dog.sigma_c = sigma_c

    @property
    def rho(self):
        return self.dog.rho

    @rho.setter
    def rho(self, rho):
        self.dog.rho = rho

    @property
    def max_S(self):
        return math.floor(self.sigma_m * 5)

    @property
    def show_progress(self):
        return self.__show_progress

    def forward(self, images):
        images = torch.mean(images, dim=1, keepdim=True)
        etf = self.etf(images)

        if self.show_progress:
            _show_arrows(etf[0,:,:,:])

        b, c, y, x = images.shape

        fdog = torch.zeros_like(images)
        for i in range(self.fdog_iters):
            if i != 0:
                images = torch.min(images, fdog)
            dog = self.dog(images, etf=etf)
            if self.show_progress:
                _show_images(images[0,:,:,:])
                _show_images(dog[0,:,:,:])

            fdog = torch.zeros_like(images)
            total_weight = 0.
            for s_dir in [-1, 1]:
                indices = torch.stack(torch.meshgrid(torch.arange(0, y), torch.arange(0, x))).repeat(b, 1, 1, 1)
                points = indices.to(dtype=images.dtype)
                for s in range(0 if s_dir == 1 else 1, self.max_S + 1):
                    if s != 0:
                        p_etf = torch.stack([torch.gather(etf[:,[i],:,:].view(b, y * x), 1, (indices[:,0,:,:] * x + indices[:,1,:,:]).view(b, y * x)).view(b, y, x) for i in range(2)], dim=1)
                        points += self.delta * p_etf * s_dir
                        points[:,0,:,:] = points[:,0,:,:].clamp(min=0, max=y-1)
                        points[:,1,:,:] = points[:,1,:,:].clamp(min=0, max=x-1)
                        indices = torch.round(points).to(dtype=torch.long)
            
                    f_s = torch.gather(dog.view(b, c, y * x), 2, (indices[:,0,:,:] * x + indices[:,1,:,:]).view(b, c, y * x)).view(b, c, y, x)
                    gauss_weight = _guassian_pdf(s, self.sigma_m)
                    total_weight += gauss_weight
                    fdog += f_s * gauss_weight

            fdog /= total_weight
            if self.show_progress:
                _show_images(fdog[0,:,:,:])
            fdog = (~((fdog < 0) * (1 + torch.tanh(fdog) < self.tau))).to(dtype=images.dtype)
            if self.show_progress:
                _show_images(fdog[0,:,:,:])

        return fdog.detach()


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
    from ..datasets import FFHQ
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.ToTensor()])

    trainset = FFHQ(root='~/Data/Flickr-Face-HQ', train=False, size='images', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=False, num_workers=2)

    dataiter = iter(trainloader)
    images = dataiter.next().to(device)
    images = images
    _show_images(images[0,:,:,:])

    fdog = FDoG()
    fdog.to(device)
    out = fdog(images)
    _show_images(out[0,:,:,:])

    '''
    torchvision.utils.save_image(images[0,:,:,:], 'test_in.png')
    torchvision.utils.save_image(out[0,:,:,:], 'test_out.png')
    '''

#---------------------------------------------------------------------