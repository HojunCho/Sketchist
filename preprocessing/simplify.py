import torch.nn as nn
import torch

import subprocess

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def SimplifyNet():
	model = nn.Sequential(
		nn.Conv2d(1, 48, (5, 5), (2, 2), (2, 2)),
		nn.ReLU(),
		nn.Conv2d(48, 128, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(1024, 512, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.ConvTranspose2d(256, 256, (4, 4), (2, 2), (1, 1), (0, 0)),
		nn.ReLU(),
		nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), (0, 0)),
		nn.ReLU(),
		nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(128, 48, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.ConvTranspose2d(48, 48, (4, 4), (2, 2), (1, 1), (0, 0)),
		nn.ReLU(),
		nn.Conv2d(48, 24, (3, 3), (1, 1), (1, 1)),
		nn.ReLU(),
		nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1)),
		nn.Sigmoid(),
	)
	return model

def process_data(data):
	immean = 0.9664114577640158
	imstd = 0.0858381272736797
	w, h = data.shape[2], data.shape[3]
	pw = 8 - (w % 8) if w % 8 != 0 else 0
	ph = 8 - (h % 8) if h % 8 != 0 else 0
	data = ((data - immean) / imstd)
	if pw != 0 or ph != 0:
		data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data)
	return data

class Simplify:
	def __init__(self):
		self.model = SimplifyNet()
		subprocess.run(['chmod', '700', 'preparation/download_simplify_model.sh'])
		subprocess.run(['preparation/download_simplify_model.sh'])
		self.model.load_state_dict(torch.load('Data/simplify_weight.pth'))
		self.model.to(device)
		self.model.eval()

	def __call__(self, data): # B C H W
		resized = process_data(data).to(device)
		return self.model(resized).detach()

#######################################
# Just for testing
#######################################
if __name__ == "__main__":
	# First, call and load the model.
	# NOTE: Do not call this several times.
	simplify = Simplify()
	simplify.model.load_state_dict(torch.load('./model_gan.pth'))

	# Prepare the input.
	# NOTE: It should have "1" channel
	x = torch.randn(10, 1, 20, 25)

	# Get output
	b = simplify(x)

	# Check the output.
	# In this case, it's (10, 1, 24, 32)
	print(b.shape)


