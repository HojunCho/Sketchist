from typing import List

import numpy as np  # type: ignore
import torch
import torchvision  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from PIL import Image, ImageFilter  # type: ignore
from torchvision import transforms


class XDoG:
    def __init__(
        self, gamma: float, phi: int, epsilon: float, k: float, sigma: float
    ) -> None:
        self.gamma = gamma
        self.phi = phi
        self.epsilon = epsilon
        self.k = k
        self.sigma = sigma

        self.to_pil = transforms.ToPILImage()

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """perform XDoG given a tensor with batch size"""

        device = t.device
        t = t.cpu()

        # go through the batch and convert all images, stack them for the output
        r: List[torch.Tensor] = []
        for i in range(t.shape[0]):
            pil_img = self.to_pil(t[i].T).convert("LA")  # type: ignore
            out = xdog(pil_img, self.gamma, self.phi, self.epsilon, self.k, self.sigma)

            o = torch.from_numpy(out).float()
            r.append(o)

        return torch.stack([v for v in r]).to(device)


def xdog_from_path(
    img_path: str, gamma: float, phi: int, epsilon: float, k: float, sigma: float
) -> None:
    raw_image = Image.open(img_path).convert("LA")
    arr = xdog(raw_image, gamma, phi, epsilon, k, sigma)

    plt.imshow(arr, cmap="gray", vmin=0, vmax=1)
    plt.show()


def xdog(
    img: Image, gamma: float, phi: int, epsilon: float, k: float, sigma: float
) -> np.array:
    """load the image at the given path and return a numpy array with pixel data"""

    # apply gaussian blurs
    g_filtered_img_1 = img.filter(ImageFilter.GaussianBlur(sigma))
    g_filtered_img_2 = img.filter(ImageFilter.GaussianBlur(sigma * k))

    # convert to numpy 128 x 128 x 2
    img_1 = np.array(g_filtered_img_1.getdata())
    dim = int(img_1.shape[0] ** (1 / 2))
    img_1 = img_1.reshape(dim, dim, 2)
    img_2 = np.array(g_filtered_img_2.getdata()).reshape(dim, dim, 2)

    # the last layer is only 255, so get the pixel value normalized to 1
    img_1 = img_1[:, :, 0] / 255
    img_2 = img_2[:, :, 0] / 255

    # plt.imshow(img_1, cmap="gray", vmin=0, vmax=1)
    # plt.show()

    xdog_img = img_1 - (gamma * img_2)

    # Extended difference of gaussians
    for i in range(xdog_img.shape[0]):
        for j in range(xdog_img.shape[1]):
            if xdog_img[i, j] < epsilon:
                xdog_img[i, j] = 1.0
            else:
                xdog_img[i, j] = 1.0 + np.tanh(phi * xdog_img[i, j])

    # plt.imshow(xdog_img, cmap="gray", vmin=0, vmax=1)
    # plt.show()

    # take mean of XDoG Filtered image to use in thresholding operation
    mean_val = np.mean(xdog_img)

    # thresholding
    for i in range(xdog_img.shape[0]):
        for j in range(xdog_img.shape[1]):
            if xdog_img[i, j] <= mean_val:
                xdog_img[i, j] = 0.0
            else:
                xdog_img[i, j] = 1.0

    return xdog_img


GAMMA = 0.97
PHI = 200
EPSILON = -0.1
K = 1.6
SIGMA = 0.8

if __name__ == "__main__":
    # xdog_from_path("./test.png", GAMMA, PHI, EPSILON, K, SIGMA)

    img1 = plt.imread("./test.png")
    img2 = plt.imread("./test.png")

    b = np.stack((img1, img2))
    batched = torch.from_numpy(b).cuda()
    xform = XDoG(GAMMA, PHI, EPSILON, K, SIGMA)

    out = xform(batched)
    for i in range(out.shape[0]):
        plt.imshow(out[i].T.cpu().numpy())  # type: ignore
        plt.show()
