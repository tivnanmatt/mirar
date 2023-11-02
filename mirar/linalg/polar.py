
import torch

from .core import RealMatrixOperator
from .interp import NearestNeighborInterpolator, BilinearInterpolator, LanczosInterpolator 


class PolarCoordinateResampler(RealMatrixOperator):
    def __init__(self, input_shape, theta_values, radius_values, row_center=None, col_center=None, interpolator=None):
        """
        This class implements a polar coordinate transformation that can be used in a PyTorch model.

        parameters:
            num_row: int
                The number of rows of the input image.
            num_col: int
                The number of columns of the input image.
            theta_values: torch.Tensor of shape [num_theta]
                The theta values, in radians, of the polar grid.
            radius_values: torch.Tensor of shape [num_radius]
                The radius values, in pixels, of the polar grid.
        """
        output_shape = (len(theta_values), len(radius_values))
        super(PolarCoordinateResampler, self).__init__(input_shape, output_shape)
                
        assert len(input_shape) == 2, "The input shape must be a tuple of length 2."
        
        # Store the number of rows and columns
        self.num_row = input_shape[0]
        self.num_col = input_shape[1]

        # Store the theta and radius values
        self.theta_values = theta_values
        self.radius_values = radius_values

        # Store the number of theta and radius values
        self.num_theta = len(theta_values)
        self.num_radius = len(radius_values)

        # Calculate the center of the image so that it works for even and odd dimensions
        if row_center is None:
            row_center = (self.num_row - 1) // 2
        if col_center is None:
            col_center = (self.num_col - 1) // 2

        # Create a meshgrid of theta and radius values
        theta_mesh, radius_mesh = torch.meshgrid(theta_values, radius_values)

        # Convert meshgrid to row, col coordinates
        #   where col is the horizontal axis, increasing from left to right
        #   and row is the vertical axis, increasing from top to bottom
        x_mesh = radius_mesh * torch.cos(theta_mesh)
        y_mesh = radius_mesh * torch.sin(theta_mesh)
        row_mesh = row_center - y_mesh + 1
        col_mesh = col_center + x_mesh + 1

        # Flatten and stack to get interp_points with shape [num_points, 2]
        interp_points = torch.stack((row_mesh.flatten(), col_mesh.flatten()), dim=1)
        
        if interpolator is None:
            interpolator = 'lanczos'
        
        assert interpolator in ['nearest', 'bilinear','lanczos'], "The interpolator must be one of 'nearest', 'bilinear', or 'lanczos'."

        if interpolator == 'nearest':
            self.interpolator = NearestNeighborInterpolator(self.num_row, self.num_col, interp_points)
        elif interpolator == 'bilinear':
            self.interpolator = BilinearInterpolator(self.num_row, self.num_col, interp_points)
        elif interpolator == 'lanczos':
            self.interpolator = LanczosInterpolator(self.num_row, self.num_col, interp_points, kernel_size=5)

        # Store the interp points
        self.interp_points = interp_points

        # Store shape for reshaping in forward method
        self.theta_mesh = theta_mesh
        self.radius_mesh = radius_mesh

    def forward(self, x):
        """
        This method implements the forward pass of the polar coordinate transformation.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, num_row, num_col]
                The input image to which the polar coordinate transformation should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, num_theta, num_radius]
                The result of applying the polar coordinate transformation to the input image.
        """

        batch_size, num_channel, num_row, num_col = x.shape

        # Assert that the number of rows and columns is correct
        assert num_row == self.num_row, "The number of rows of x does not match the number of rows of the image."
        assert num_col == self.num_col, "The number of columns of x does not match the number of columns of the image."

        # Interpolate the values
        interpolated = self.interpolator(x)
        
        # Reshape to the original theta, r grid
        result = interpolated.view(*interpolated.shape[:2], *self.theta_mesh.shape)
        
        return result
    
    def transpose(self, y):
        """
        This method implements the transpose pass of the polar coordinate transformation.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, num_theta, num_radius]
                The input image to which the adjoint polar coordinate transformation should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, num_row, num_col]
                The result of applying the adjoint polar coordinate transformation to the input image.
        """

        batch_size, num_channel, num_theta, num_radius = y.shape

        # Assert that the number of rows and columns is correct
        assert num_theta == self.theta_mesh.shape[0], "The number of rows of x does not match the number of rows of the image."
        assert num_radius == self.radius_mesh.shape[1], "The number of columns of x does not match the number of columns of the image."

        # Flatten the polar image
        flattened = y.view(*y.shape[:2], -1)
        
        # Use the adjoint method of the interpolator
        result = self.interpolator.transpose(flattened)
        
        return result

    # def pseudo_inverse(self, x, **kwargs):
    #     """
    #     This method implements the inverse of the polar coordinate transformation.

    #     parameters:
    #         x: torch.Tensor of shape [batch_size, num_channel, num_theta, num_radius]
    #             The input image to which the inverse polar coordinate transformation should be applied.
    #     returns:
    #         result: torch.Tensor of shape [batch_size, num_channel, num_row, num_col]
    #     """

    #     batch_size, num_channel, num_theta, num_radius = x.shape

    #     # Assert that the number of rows and columns is correct
    #     assert num_theta == self.theta_mesh.shape[0], "The number of rows of x does not match the number of rows of the image."
    #     assert num_radius == self.radius_mesh.shape[1], "The number of columns of x does not match the number of columns of the image."

    #     # Flatten the polar image
    #     flattened = x.view(*x.shape[:2], -1)
        
    #     # Use the inverse method of the interpolator
    #     result = self.interpolator.pseudo_inverse(flattened, **kwargs)
        
    #     return result
    
    def to(self, *args, **kwargs):
        """Override the to method to ensure that the interpolator is moved to the new device."""
        self.interpolator = self.interpolator.to(*args, **kwargs)
        return super().to(*args, **kwargs)