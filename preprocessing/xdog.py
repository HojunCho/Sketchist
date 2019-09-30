import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

GAMMA = 0.95
PHI = 200
EPSILON = -0.1
K = 1.6
SIGMA = 0.8
DIM = 128


def xdog(img_path: str) -> np.array:
    """load the image at the given path and return a numpy array with pixel data"""

    raw_image = Image.open(img_path).convert("LA")

    # apply gaussian blurs
    g_filtered_img_1 = raw_image.filter(ImageFilter.GaussianBlur(SIGMA))
    g_filtered_img_2 = raw_image.filter(ImageFilter.GaussianBlur(SIGMA * K))

    # convert to numpy 128 x 128 x 2
    img_1 = np.array(g_filtered_img_1.getdata()).reshape(DIM, DIM, 2)
    img_2 = np.array(g_filtered_img_2.getdata()).reshape(DIM, DIM, 2)

    # the last layer is only 255, so get the pixel value normalized to 1
    img_1 = img_1[:, :, 0] / 255
    img_2 = img_2[:, :, 0] / 255

    plt.imshow(img_1, cmap="gray", vmin=0, vmax=1)
    plt.show()

    xdog_img = img_1 - (GAMMA * img_2)

    # Extended difference of gaussians
    for i in range(xdog_img.shape[0]):
        for j in range(xdog_img.shape[1]):
            if xdog_img[i, j] < EPSILON:
                xdog_img[i, j] = 1.0
            else:
                xdog_img[i, j] = 1.0 + np.tanh(PHI * xdog_img[i, j])

    plt.imshow(xdog_img, cmap="gray", vmin=0, vmax=1)
    plt.show()

    # take mean of XDoG Filtered image to use in thresholding operation
    mean_val = np.mean(xdog_img)

    # thresholding
    for i in range(xdog_img.shape[0]):
        for j in range(xdog_img.shape[1]):
            if xdog_img[i, j] <= mean_val:
                xdog_img[i, j] = 0.0
            else:
                xdog_img[i, j] = 1.0

    plt.imshow(xdog_img, cmap="gray", vmin=0, vmax=1)
    plt.show()

    return xdog_img


xdog("./test.png")
