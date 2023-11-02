import torch
from ..linalg.core import CompositeMatrixOperator
from ..linalg.core import RealMatrixOperator
from ..linalg.permute import Pad
from ..linalg.fourier import FourierTransform
from ..linalg.polar import PolarCoordinateResampler
from ..linalg.sparse import RowSparseMatrixOperator
from ..linalg.misc import RealPart

pi = 3.1415927410125732

class CTProjector_FourierSliceTheorem(CompositeMatrixOperator, RealMatrixOperator):
    def __init__(self, input_shape, num_fourier_angular_samples, num_fourier_radial_samples, theta_values=None, radius_values=None, interpolator=None, pad_factor=1):

        if theta_values is None:
            theta_values = torch.linspace(0, 2 * pi, num_fourier_angular_samples)
        if radius_values is None:
            # we use  zero padding so this is the spacing in resulting fourier space
            sample_spacing = 1.5*(1 + 2*pad_factor)*max(input_shape)/num_fourier_radial_samples 

            # Create two separate tensors for the negative and positive ranges
            negative_samples = torch.arange(start=-sample_spacing * (num_fourier_radial_samples//2), end=-sample_spacing, step=sample_spacing)
            positive_samples = torch.arange(start=0, end=sample_spacing * (num_fourier_radial_samples//2), step=sample_spacing)

            # Concatenate the tensors to get the final radius tensor
            radius_values = torch.cat([negative_samples, positive_samples], dim=0)
            
            # Store parameters
            self.num_row = input_shape[0]
            self.num_col = input_shape[1]
            self.num_fourier_angular_samples = num_fourier_angular_samples
            self.num_fourier_radial_samples = num_fourier_radial_samples
            self.theta_values = theta_values
            self.radius_values = radius_values

            # Create a pad operator to zero-pad the input image
            pad = Pad(input_shape, (pad_factor*input_shape[0], pad_factor*input_shape[0], pad_factor*input_shape[1], pad_factor*input_shape[1]))

            # Create 2D Fourier transform module
            fourier_transform_2d = FourierTransform((input_shape[0]*(1 + 2*pad_factor), input_shape[1]*(1 + 2*pad_factor)), dim=(-2, -1))

            # Create Polar Coordinate Transformation module
            polar_transform = PolarCoordinateResampler((input_shape[0]*(1 + 2*pad_factor), input_shape[1]*(1 + 2*pad_factor)), theta_values, radius_values, interpolator=interpolator)

            # Create 1D Fourier transform module
            fourier_transform_1d = FourierTransform((num_fourier_radial_samples,), dim=(-1,))

            # Create a real part module
            real_part_1 = RealPart(input_shape)
            real_part_2 = RealPart((num_fourier_angular_samples, num_fourier_radial_samples))

            operators = [real_part_1, pad, fourier_transform_2d, polar_transform, fourier_transform_1d.conjugate_transpose_MatrixOperator(), real_part_2]

            super(CTProjector_FourierSliceTheorem, self).__init__(operators)

class CTProjector_ParallelBeam2D(RealMatrixOperator):

    def __init__(self, Nx, Ny, dx, dy, Nu, du, theta, verbose=False, device='cpu'):
        input_shape = (Ny, Nx)
        output_shape = (len(theta), Nu)
        super(CTProjector_ParallelBeam2D, self).__init__(input_shape, output_shape)

        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.Nu = Nu
        self.du = du
        self.theta = theta
        self.Ntheta = len(theta)
        self.verbose = verbose
        self._device = device

    def forward(self, x):

        batch_size, num_channel = x.shape[:2]

        # This will loop through the views, creating a RowSparseMatrixOperator for each view
        result = torch.zeros([batch_size,num_channel,self.Ntheta, self.Nu], dtype=torch.float32).to(self._device)
        for iTheta, theta_i in enumerate(self.theta):
            if self.verbose:
                print('Forward Projecting View : ', iTheta+1, ' of ', self.Ntheta)
            
            # Compute the overlap between the voxel trapezoidal footprint and the pixel only for nonzero pixels. 
            # Return pixel indices and nonzero areas
            pixel_index, area_between_pixel_trapezoidal_footprint = self._system_response(theta_i, *torch.meshgrid(
                torch.linspace(-(self.Ny-1)/2, (self.Ny-1)/2, self.Ny, device=self._device) * self.dy,
                torch.linspace(-(self.Nx-1)/2, (self.Nx-1)/2, self.Nx, device=self._device) * self.dx,
                indexing='ij'))
            
            area_between_pixel_trapezoidal_footprint[pixel_index<0] = 0
            area_between_pixel_trapezoidal_footprint[pixel_index>=self.Nu] = 0
            pixel_index[pixel_index<0] = 0
            pixel_index[pixel_index>=self.Nu] = self.Nu-1
            
            operator = RowSparseMatrixOperator(self.input_shape, (self.Nu,), pixel_index, area_between_pixel_trapezoidal_footprint)
            result[:,:,iTheta] = operator.forward(x).squeeze()
        
        return result

    def transpose(self, y):

        batch_size, num_channel = y.shape[:2]

        # This will loop through the views, creating a RowSparseMatrixOperator for each view
        result = torch.zeros([batch_size,num_channel,self.Ny, self.Nx], dtype=torch.float32).to(self._device)
        for iTheta, theta_i in enumerate(self.theta):
            if self.verbose:
                print('Back Projecting View : ', iTheta+1, ' of ', self.Ntheta)
            
            # Compute the overlap between the voxel trapezoidal footprint and the pixel only for nonzero pixels. 
            # Return pixel indices and nonzero areas
            pixel_index, area_between_pixel_trapezoidal_footprint = self._system_response(theta_i, *torch.meshgrid(
                torch.linspace(-(self.Ny-1)/2, (self.Ny-1)/2, self.Ny, device=self._device) * self.dy,
                torch.linspace(-(self.Nx-1)/2, (self.Nx-1)/2, self.Nx, device=self._device) * self.dx,
                indexing='ij'))
            
            area_between_pixel_trapezoidal_footprint[pixel_index<0] = 0
            area_between_pixel_trapezoidal_footprint[pixel_index>=self.Nu] = 0
            pixel_index[pixel_index<0] = 0
            pixel_index[pixel_index>=self.Nu] = self.Nu-1

            operator = RowSparseMatrixOperator(self.input_shape, (self.Nu,), pixel_index, area_between_pixel_trapezoidal_footprint)
            result[:,:] += operator.transpose(y[:,:,iTheta]).squeeze()
        
        return result
    
    def _convert_u_to_projection_index(self, u):
        return (u/self.du) + (self.Nu-1)/2.0

    def _convert_projection_index_to_u(self, projection_index):
        return (projection_index - (self.Nu-1)/2.0)*self.du

    def _system_response(self, theta_i, y2d, x2d):

        # compute the projection
        theta_i = torch.tensor(theta_i)

        # absolute value of the sine and cosine of the angle
        abs_cos_theta = torch.abs(torch.cos(theta_i))
        abs_sin_theta = torch.abs(torch.sin(theta_i))

        # height of the trapezoid
        h = torch.minimum(self.dy/abs_cos_theta, self.dx/abs_sin_theta)
        # base 1 of the trapezoid
        b1 = torch.abs(self.dy*abs_sin_theta - self.dx*abs_cos_theta)
        # base 2 of the trapezoid
        b2 = torch.abs(self.dy*abs_sin_theta + self.dx*abs_cos_theta)

        # below depends on x and y, above depends only on theta

        # center of the trapezoid
        u0 = (x2d*torch.cos(theta_i) + y2d*torch.sin(theta_i))
        # left edge of the trapezoid base 2 
        u1 = u0 - b2/2
        # left edge of the trapezoid base 1
        u2 = u0 - b1/2
        # right edge of the trapezoid base 1
        u3 = u0 + b1/2
        # right edge of the trapezoid base 2
        u4 = u0 + b2/2

        # compute the index of the projection
        u1_index = self._convert_u_to_projection_index(u1)
        u4_index = self._convert_u_to_projection_index(u4)
        num_nonzero_pixels = int(torch.max(torch.ceil(u4_index)-torch.floor(u1_index))) + 1
        pixel_index = torch.zeros([num_nonzero_pixels, x2d.shape[0], x2d.shape[1]], dtype=torch.long).to(self._device)
        area_between_pixel_trapezoidal_footprint = torch.zeros([num_nonzero_pixels, x2d.shape[0], x2d.shape[1]], dtype=torch.float).to(self._device)

        for iPixel in range(num_nonzero_pixels):
            # get the index of the pixel of interest
            pixel_index[iPixel] = torch.floor(u1_index).long() + iPixel
            # convert index to u coordinate
            u = self._convert_projection_index_to_u(pixel_index[iPixel])
            # area of the left side of the trapezoid
            u_A = torch.maximum(u1,u-self.du/2)
            u_B = torch.minimum(u2,u+self.du/2)
            area_between_pixel_trapezoidal_footprint[iPixel] += (u_B>u_A)*(h/(2*(u2-u1 + (u1==u2))))*((u_B-u1)**2.0 - (u_A-u1)**2.0)
            # area of the center of the trapezoid
            u_A = torch.maximum(u2,u-self.du/2)
            u_B = torch.minimum(u3,u+self.du/2)
            area_between_pixel_trapezoidal_footprint[iPixel] += (u_B>u_A)*h*(u_B-u_A)
            # area of the right side of the trapezoid
            u_A = torch.maximum(u3,u-self.du/2)
            u_B = torch.minimum(u4,u+self.du/2)
            area_between_pixel_trapezoidal_footprint[iPixel] += (u_B>u_A)*(h/(2*(u4-u3+ (u3==u4))))*((u_A-u4)**2.0 - (u_B-u4)**2.0)
        return pixel_index, area_between_pixel_trapezoidal_footprint 
    
    def to(self, device):
        self._device = device
        return self
    



class CTProjector_ExternalMatched(RealMatrixOperator):
    def __init__(self, forward_projector_fun, back_projector_fun, input_shape, output_shape):
        super(CTProjector_ExternalMatched, self).__init__(input_shape, output_shape)
        self.forward_projector_fun = forward_projector_fun
        self.back_projector_fun = back_projector_fun

    def forward(self, x):
        return self.forward_projector_fun(x) 
       
    def transpose(self, y):
        return self.back_projector_fun(y)
