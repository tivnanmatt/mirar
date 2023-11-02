
import torch

import matplotlib
from matplotlib import pyplot as plt

plt.figure()
plt.show(block=True)

# define pi. We avoid using numpy to avoid conflicts with pytorch
pi = 3.1415927410125732

# get the path to the directory containing this file and add the previous directory to the path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# now we can import from torch_imaging
from mirar.physics.ct_projector import CTProjector_FourierSliceTheorem

# if the GPU is available, use it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Testing the FourierSliceForwardProjector module

# now make a (256, 256) x,y coordinate meshgrid
x_mesh, y_mesh = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))

# define a spiral
theta = torch.linspace(0, 10 * pi, 1000)
radius = torch.linspace(0, 1, 1000)

# convert to x,y coordinates
x = radius * torch.cos(theta)
y = radius * torch.sin(theta)

# make a mask of those x,y coordinates
spiral = torch.zeros_like(x_mesh, dtype=torch.float32, device=device)
for i in range(len(x)):
    spiral[(x_mesh - x[i])**2 + (y_mesh - y[i])**2 < .001] = 1

# show the mask
plt.imshow(spiral.cpu())
plt.title('Spiral Mask')
plt.show()
    
# define the number of Fourier domain polar samples to use
num_fourier_angular_samples = 1440
num_fourier_radial_samples = 1024

# we are using an even number of samples, 
forward_projector = CTProjector_FourierSliceTheorem(    spiral.shape, 
                                                        num_fourier_angular_samples, 
                                                        num_fourier_radial_samples, 
                                                        interpolator='lanczos')
# move it to the GPU if available
forward_projector = forward_projector.to(device)

# Apply the forward projector
y = forward_projector(spiral.unsqueeze(0).unsqueeze(0))

# Show the result
plt.figure()
plt.imshow(y[0, 0].cpu().real, aspect='auto')
plt.title('Forward Projection of Spiral Mask')
plt.show(block=True)

# Apply the backprojector
x_bp = forward_projector.transpose(y)

# Show the result
plt.figure()
plt.imshow(x_bp[0, 0].cpu().real, aspect='auto')
plt.title('Back-Projected Spiral Mask')
plt.show(block=True)

# Apply the inverse projector
x_inv = forward_projector.pseudo_inverse(y, max_iter=40, verbose=True)

# Show the result
plt.figure()
plt.imshow(x_inv[0, 0].cpu().real, aspect='auto')
plt.title('Reconstructed Spiral Mask')
plt.show(block=True)