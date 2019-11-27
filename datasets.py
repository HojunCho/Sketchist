from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import os
from preparation.download_ffhq import run

from preprocessing import xdog
from preprocessing import fdog
from preprocessing import simplify
from preprocessing import removal

# ---------------------------------------------------------------------
# Import packages for testing

import torch
import torchvision

import matplotlib.pyplot as plt

import numpy as np

# ---------------------------------------------------------------------

class FFHQ(ImageFolder):
    def __init__(self, root='Data', train=True, size='thumbs', stats=False, transform=None):
        print('Initializing FFHQ dataset: ' + size + (' training' if train else ' validation') + ' data')

        dir_path = os.path.join(os.path.expanduser(root))
        os.makedirs(dir_path, exist_ok=True)
        prev_dir = os.getcwd()
        os.chdir(dir_path)
        run(tasks=['json', size] + (['stats'] if stats else []), train=train)
        os.chdir(prev_dir)

        dir_path = os.path.join(dir_path, 'train' if train else 'val')
        if size == 'thumbs':
            dir_path = os.path.join(dir_path, 'thumbnails128x128')
        elif size == 'images':
            dir_path = os.path.join(dir_path, 'images1024x1024')
        elif size == 'wilds':
            dir_path = os.path.join(dir_path, 'in-the-wild-images')
        super(FFHQ, self).__init__(dir_path, transform=transform)

        print('Done!')

    def __getitem__(self, index):
        return super(FFHQ, self).__getitem__(index)[0]


class SketchDataLoader(DataLoader):
    def __init__(self, root='Data', sketch_type='XDoG', train=True, size=64, size_from='thumbs', device='cpu', **kwargs):
        self.device = device
        transform = transforms.Compose(
            [transforms.Resize(size=size),
             transforms.ToTensor()])

        if sketch_type == 'XDoG':
            self.edge_detector = xdog.XDoG(gamma=xdog.GAMMA,
                                           phi=xdog.PHI,
                                           epsilon=xdog.EPSILON,
                                           k=xdog.K,
                                           sigma=xdog.SIGMA)

        elif sketch_type == 'FDoG':
            if size == 64:
                fdog_parameters = {
                    'mu': 2,
                    'sigma_c': 1.0,
                    'sigma_m': 2.0
                }
            if size == 128:
                fdog_parameters = {
                    'mu': 3,
                    'sigma_c': 1.0,
                    'sigma_m': 3.0
                }
            elif size == 1024:
                fdog_parameters = {
                    'mu': 15,
                    'sigma_c': 2.0,
                    'sigma_m': 5.0
                }
            else:
                raise ValueError('Invalid image size')
            self.edge_detector = fdog.FDoG(**fdog_parameters).to(device=device)

        else:
            raise ValueError('Bad sketch method name')

        self.removal = removal.Removal(device=device)
        self.simplifier = simplify.Simplify(device=device)
        kwargs['dataset'] = FFHQ(root=root, train=train, size=size_from, stats=False, transform=transform)
        super(SketchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        iterator = super(SketchDataLoader, self).__iter__()

        def _iter_wrapper(iterator):
            for real_image in iterator:
                foreground = self.removal((real_image * 255).to(torch.uint8)).float() / 255.
                edges = self.edge_detector(foreground)
                sketch = self.simplifier(edges.to(self.device)).repeat([1, 3, 1, 1])
                yield (torch.cat([sketch, real_image.to(self.device)], dim=3) - .5) / .5

        return _iter_wrapper(iterator)


# ---------------------------------------------------------------------
# Test code

if __name__ == "__main__":
    trainloader = SketchDataLoader(root='~/Data/Datasets/Flickr-Face-HQ', train=False, sketch_type='XDoG',
                                   size=64, size_from='thumbs', batch_size=4, shuffle=True, num_workers=2)

    dataiter = iter(trainloader)
    images = next(dataiter)
    print(images.shape)

    # show images
    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# ---------------------------------------------------------------------
