from datetime import datetime, timedelta
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
        t = t.permute(0, 3, 2, 1).cpu()

        # go through the batch and convert all images, stack them for the output
        r: List[torch.Tensor] = []
        for i in range(t.shape[0]):
            pil_img = self.to_pil(t[i].T).convert("L")  # type: ignore
            out = xdog(pil_img, self.gamma, self.phi, self.epsilon, self.k, self.sigma)

            o = torch.from_numpy(out).float()
            r.append(o)

        return torch.stack([v for v in r]).to(device)


def xdog_from_path(
    img_path: str, gamma: float, phi: int, epsilon: float, k: float, sigma: float
) -> None:
    raw_image = Image.open(img_path).convert("L")
    arr = xdog(raw_image, gamma, phi, epsilon, k, sigma)

    plt.imshow(arr.T, vmin=0, vmax=1)
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
    img_1 = img_1.reshape(dim, dim, 1)
    img_2 = np.array(g_filtered_img_2.getdata()).reshape(dim, dim, 1)

    # the last layer is only 255, so get the pixel value normalized to 1
    img_1 = img_1[:, :, 0] / 255
    img_2 = img_2[:, :, 0] / 255

    # plt.imshow(img_1, cmap="gray", vmin=0, vmax=1)
    # plt.show()

    xdog_img = img_1 - (gamma * img_2)
    xdog_img = np.where(xdog_img < epsilon, 1.0, 1.0 + np.tanh(phi * xdog_img))

    # thresholding
    mean_val = np.mean(xdog_img)
    xdog_img = np.where(xdog_img <= mean_val, 0.0, 1.0)

    return np.expand_dims(xdog_img, axis=0)


GAMMA = 0.95
PHI = 200
EPSILON = 0.015
K = 1.2
SIGMA = 0.9

if __name__ == "__main__":
    # xdog_from_path("./test.png", GAMMA, PHI, EPSILON, K, SIGMA)

    img1 = plt.imread("./test.png")
    img2 = plt.imread("./test.png")

    b = np.stack((img1, img2))
    batched = torch.from_numpy(np.transpose(b,axes=[0, 3, 1, 2]))
    print(batched.shape)

    xform = XDoG(GAMMA, PHI, EPSILON, K, SIGMA)
    times = []
    for i in range(10):
        print(i)
        before = datetime.now()
        out = xform(batched)
        after = datetime.now()
        times.append(after - before)

    print("avg time per image: ", sum(times, timedelta(0)) / len(times) / 2)

    print(out.shape)
    for i in range(out.shape[0]):
        plt.imshow(out[i].T.cpu().numpy())  # type: ignore
        plt.show()
