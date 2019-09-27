from torchvision.datasets import ImageFolder

import os
from download_ffhq import run

#---------------------------------------------------------------------
# Import packages for testing

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np

#---------------------------------------------------------------------

class FFHQ(ImageFolder):
    def __init__(self, root='Data', train=True, size='thumbs', stats=False, transform=None, target_transform=None):
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
        super(FFHQ, self).__init__(dir_path, transform=transform, target_transform=target_transform)

        print('Done!')
    
    def __getitem__(self, index):
        return super(FFHQ, self).__getitem__(index)[0]

#---------------------------------------------------------------------
# Test code

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = FFHQ(root='~/Data/Flickr-Face-HQ', train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    dataiter = iter(trainloader)
    images = dataiter.next()

    # show images
    img = torchvision.utils.make_grid(images)
    img = img /2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#---------------------------------------------------------------------