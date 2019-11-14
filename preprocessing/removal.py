from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import torchvision.transforms as T

def decode_segmap(image, source, nc=21):
  label_colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0),
    (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0),
    (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)

  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]

  rgb = np.stack([r, g, b], axis=2)

  # Load the foreground input image
  # foreground = cv2.imread(source)
  foreground = source
  # and resize image to match shape of R-band in RGB output map
  #foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
  foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))
  # Create a background array to hold white pixels
  # with the same size as RGB output map
  background = 255 * np.ones_like(rgb).astype(np.uint8)
  # Convert uint8 to float
  foreground = foreground.astype(float)
  background = background.astype(float)
  # Create a binary mask of the RGB output map using the threshold value 0
  th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)
  # Apply a slight blur to the mask to soften edges
  alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
  # Normalize the alpha mask to keep intensity between 0 and 1
  alpha = alpha.astype(float) / 255
  # Multiply the foreground with the alpha matte
  foreground = cv2.multiply(alpha, foreground)
  # Multiply the background with ( 1 - alpha )
  background = cv2.multiply(1.0 - alpha, background)
  # Add the masked foreground and background
  outImage = cv2.add(foreground, background)
  # Return a normalized output image for display
  return outImage / 255

def segment(net, data, show_orig=True, dev='cuda'):
  #if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  trf = T.Compose([T.ToTensor(),
                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  rgbs = []
  for i in range(data.shape[0]):
    inp = trf(data[i]).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    rgb = decode_segmap(om, data[i])
    rgb = rgb*255
    rgbs.append(rgb)

  rgbs = np.stack(rgbs, axis=0)
  return rgbs


class Removal:
  def __init__(self, device):
    self.device = device
    self.model = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

  def __call__(self, data): # Input path
    return segment(self.model, data, show_orig=False)


if __name__ == "__main__":
  a = cv2.imread('./00000.png')
  b = cv2.imread('./00003.png')
  test_input = np.stack([a, b], axis=0) # B X W X H X C

  removal = Removal('cuda')
  output = removal(test_input) # B X W X H X C

  a_o = output[0]
  cv2.imwrite(('a_o.png'), a_o)
  b_o = output[1]
  cv2.imwrite(('b_o.png'), b_o)

