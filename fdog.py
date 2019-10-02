import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from datasets import FFHQ

class Sobel(nn.Module):

    def __init__(self):
        super(Sobel, self).__init__()        
        kernel = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ])
        self.kernel = torch.stack([kernel, kernel.transpose(0, 1)])

    def forward(self, inputs):
        kernel = self.kernel.to(device=inputs.device, dtype=inputs.dtype)
        kernel = kernel.unsqueeze(dim=1)

        return F.conv2d(inputs, kernel, padding=1)


class ETF(nn.Module):

    def __init__(self, mu=5, eta=1., iterations=3):
        super(ETF, self).__init__()
        self.mu = mu
        self.eta = eta
        self.iterations = iterations

        self.sobel = Sobel()
        self.padding = nn.ZeroPad2d(mu)

    def forward(self, inputs):
        sobel = self.sobel(inputs)
        sobel_mag = torch.norm(sobel, p=None, dim=1, keepdim=True)
        sobel_mag /= torch.max(sobel_mag)
        sobel_mag_padded = self.padding(sobel_mag)
        sobel_X = sobel_mag

        tang = sobel.flip(dims=[1]) * torch.tensor([-1., 1.], device=inputs.device, dtype=inputs.dtype).view(1, 2, 1, 1)
        tang_mag = torch.norm(tang, p=None, dim=1, keepdim=True)
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

                    tang += tang_Y * (torch.tanh(self.eta * (sobel_Y - sobel_X)) + 1) * (tang_X * tang_Y).sum(dim=1, keepdim=True) / 2.

            tang_mag = torch.norm(tang, p=None, dim=1, keepdim=True)
            tang_mag[tang_mag == 0] = 1
            tang /= tang_mag

        return tang


#---------------------------------------------------------------------
# Test code
def _show_images(images):
    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

def _show_arrows(vector_map):
    #vector_map = torchvision.utils.make_grid(vector_map)
    vector_map = vector_map[0,:,:,:]
    np_map = vector_map.numpy()

    c, x, y = np_map.shape
    X, Y = np.meshgrid(np.arange(0, x), np.arange(0, y))
    U = np_map[0,:,:]
    V = np_map[1,:,:]
    Q = plt.quiver(X, Y, U, V, units='inches')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.ToTensor()])

    trainset = FFHQ(root='~/Data/Flickr-Face-HQ', train=False, size='thumbs', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    dataiter = iter(trainloader)
    images = dataiter.next().to(device)

    _show_images(images)
    etf = ETF(mu=5, iterations=10)
    etf.to(device)
    out = etf(images)
    
    _show_arrows(out)

#---------------------------------------------------------------------