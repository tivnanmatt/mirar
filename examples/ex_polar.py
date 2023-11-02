
import torch
from matplotlib import pyplot as plt

# define pi. We avoid using numpy to avoid conflicts with pytorch
pi = 3.1415927410125732

# get the path to the directory containing this file and add the previous directory to the path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mirar.linalg.polar import PolarCoordinateResampler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
# Testing the PolarCoordinateResampler module
theta = torch.linspace(0, 2 * pi, 720)
radius = torch.linspace(0, 200, 300)

cart2pol = PolarCoordinateResampler((256, 256), theta, radius, interpolator='bilinear').to(device)

# now make a (72, 72) x,y coordinate meshgrid
x_mesh, y_mesh = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))

# define a spiral
theta = torch.linspace(0, 10 * pi, 1000)
radius = torch.linspace(0, 1, 1000)

# convert to x,y coordinates
x = radius * torch.cos(theta)
y = radius * torch.sin(theta)

# make a mask of those x,y coordinates
mask = torch.zeros_like(x_mesh, dtype=torch.float32, device=device)
for i in range(len(x)):
    mask[(x_mesh - x[i])**2 + (y_mesh - y[i])**2 < .001] = 1

# show the mask
plt.imshow(mask.cpu())
plt.show()

# show the mask
plt.figure()
plt.imshow(mask.cpu())
plt.plot(cart2pol.interp_points[:, 1].cpu(), cart2pol.interp_points[:, 0].cpu(), 'r.')
plt.xlim([110, 145])
plt.ylim([110, 145])
plt.show(block=True)


# reshape it to a (1, 1, 72, 72) tensor
mask = mask.view(1, 1, 256, 256)
mask = mask.type(torch.float32)
# mask = mask*0 + 1

# apply the polar coordinate transformation
output = cart2pol(mask)

# output = output + torch.randn_like(output)*0.5
# show the output
plt.imshow(output[0, 0].cpu())
plt.show()

# Applying the inverse transformation
inverse_output = cart2pol.pseudo_inverse(output, max_iter=100, tol=1e-6, verbose=True)

# Show the output of the inverse transformation
plt.figure()
plt.imshow(inverse_output[0, 0].cpu(),vmin=0,vmax=1)
plt.show(block=True)




