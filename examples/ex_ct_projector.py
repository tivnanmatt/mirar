import torch
from matplotlib import pyplot as plt

# define pi. We avoid using numpy to avoid conflicts with pytorch
pi = 3.1415927410125732

# get the path to the directory containing this file and add the previous directory to the path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# now we can import the new projector
from mirar.physics.ct_projector import CTProjector_ParallelBeam2D

# if the GPU is available, use it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define parameters for the new projector
Nx, Ny = 256, 256
dx, dy = 2/255, 2/255  # Considering the linspace range from -1 to 1
Nu = 1024
du = 3/1023  # Considering the range for projection is from -1 to 1
theta = torch.linspace(0, 2*pi, 180)  # 180 projection angles from 0 to pi

# now make a (256, 256) x,y coordinate meshgrid
x_mesh, y_mesh = torch.meshgrid(torch.linspace(-1, 1, Nx), torch.linspace(-1, 1, Ny))

# define a spiral
theta_spiral = torch.linspace(0, 10 * pi, 1000)
radius = torch.linspace(0, 1, 1000)

# convert to x,y coordinates
x = radius * torch.cos(theta_spiral)
y = radius * torch.sin(theta_spiral)

# make a mask of those x,y coordinates
spiral = torch.zeros_like(x_mesh, dtype=torch.float32, device=device)
for i in range(len(x)):
    spiral[(x_mesh - x[i])**2 + (y_mesh - y[i])**2 < .001] = 1

# show the mask
plt.imshow(spiral.cpu())
plt.title('Spiral Mask')
plt.show()

# Initialize the new projector
projector = CTProjector_ParallelBeam2D(Nx, Ny, dx, dy, Nu, du, theta, verbose=False)
projector = projector.to(device)

# Apply the forward projector
y = projector.forward(spiral.unsqueeze(0).unsqueeze(0))

# Show the result
plt.figure()
plt.imshow(y.cpu()[0,0], aspect='auto')
plt.title('Forward Projection of Spiral Mask')
plt.show()

# Apply the backprojector
x_bp = projector.transpose(y)

# Show the result
plt.figure()
plt.imshow(x_bp.cpu()[0,0].cpu(), aspect='auto')
plt.title('Back-Projected Spiral Mask')
plt.show()

# Note: The CTProjector_ParallelBeam2D doesn't have a pseudo_inverse method like CTProjector_FourierSliceTheorem.
# If needed, this method can be implemented separately.
