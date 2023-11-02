import torch
from matplotlib import pyplot as plt
import numpy as np

# get the path to the directory containing this file and add the previous directory to the path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# now we can import from torch_imaging
from mirar.physics.ct_projector import CTProjector_FourierSliceTheorem
from mirar.linalg.polar import PolarCoordinateResampler

# define pi
pi = 3.1415927410125732

# Device selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. Generate a simple circular disc
x_mesh, y_mesh = torch.meshgrid(torch.linspace(-128, 128, 256), torch.linspace(-128, 128, 256))
disc = ((x_mesh-30)**2 + (y_mesh-30)**2 <= 50**2).float().to(device)
plt.imshow(disc.cpu())
plt.title("Circular Disc")
plt.show()

# 2. Define CT projector and apply forward projection
# Note: Assuming a function similar to CTProjector_FourierSliceTheorem exists for forward projection
# Forward projection function and parameters are based on the example you provided
forward_projector = CTProjector_FourierSliceTheorem(disc.shape, 1440, 1024*6, interpolator='lanczos', pad_factor=1).to(device)
sinogram = forward_projector(disc.unsqueeze(0).unsqueeze(0))
plt.imshow(sinogram[0, 0].cpu().real, aspect='auto')
plt.title("Sinogram")
plt.show()

# 3. Add random noise to the sinogram
noise = 0.1*torch.randn(1, 1, 1, sinogram.shape[3] ).to(device)  # Random Gaussian noise [1, N_detector]
sinogram_noisy = sinogram + noise
plt.imshow(sinogram_noisy[0, 0].cpu().real, aspect='auto')
plt.title("Sinogram with Noise")
plt.show()

# 4. Apply pseudo inverse reconstruction
reconstructed_image = forward_projector.pseudo_inverse(sinogram_noisy, max_iter=20, verbose=True)
plt.imshow(reconstructed_image[0, 0].cpu().real, vmin=0, vmax=2)
plt.title("Reconstructed Image with Ring Artifacts")
plt.show()

# 5. Apply polar transformation
theta = torch.linspace(0, 2 * pi, 720)
radius = torch.linspace(0, 200, 300)
cart2pol = PolarCoordinateResampler((256, 256), theta, radius, interpolator='bilinear').to(device)
polar_image = cart2pol(reconstructed_image)
plt.imshow(polar_image[0, 0].cpu())
plt.title("Polar Transformed Image")
plt.show()

# 6. Apply smoothing in the radial direction
# smoothed_polar_image = torch.nn.functional.avg_pool2d(polar_image, (3, 1))
# the above is not correct because i dont want to downsample, i want to blur in the radial direction
polar_image_blurred = torch.nn.functional.conv2d(polar_image, torch.ones(1, 1, 1, 21).to(device)/21, padding=(0, 10))
norm = torch.nn.functional.conv2d(torch.ones_like(polar_image), torch.ones(1, 1, 1, 21).to(device)/21, padding=(0, 10))
polar_image_blurred = polar_image_blurred/norm
plt.imshow(polar_image_blurred[0, 0].cpu())
plt.title("Smoothed Polar Image")
plt.show()

# 7. Apply inverse polar transformation
corrected_image = cart2pol.pseudo_inverse(polar_image_blurred)
plt.imshow(corrected_image[0, 0].cpu(), vmin=0, vmax=1)
plt.title("Corrected Image")
plt.show()