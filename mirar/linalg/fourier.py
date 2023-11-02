
import torch

from .matrix_operator import UnitaryMatrixOperator, DiagonalMatrixOperator, EigenDecomposedMatrixOperator
    
class FourierTransform(UnitaryMatrixOperator):
    def __init__(self, input_shape, dim):
        """
        This class implements a N-Dimensional Fourier transform that can be used in a PyTorch model.

        it assumes the central pixel in the image is at 0

        it returns the Fourier transform with the zero-frequency component in the center of the image
        """
        super(FourierTransform, self).__init__(input_shape)
        self.dim = dim

    def forward(self, x):
        x_ifftshift = torch.fft.ifftshift(x, dim=self.dim)
        x_fft = torch.fft.fftn(x_ifftshift, dim=self.dim, norm="ortho")
        x_fftshift = torch.fft.fftshift(x_fft, dim=self.dim)
        return x_fftshift
    
    def transpose(self, y):
        return torch.conj(self.conjugate_transpose( torch.conj(y)))

    def conjugate_transpose(self, y):
        y_ifftshift = torch.fft.ifftshift(y, dim=self.dim)
        y_ifft = torch.fft.ifftn(y_ifftshift, dim=self.dim, norm="ortho")
        y_fftshift = torch.fft.fftshift(y_ifft, dim=self.dim)
        return y_fftshift

class FourierFilter(EigenDecomposedMatrixOperator):
    def __init__(self, input_shape, filter, dim):
        """
        This class implements a 2D Fourier filter that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0)

        it returns the Fourier filter applied to the input. 
        """
        eigenvectors = FourierTransform(input_shape, dim=dim)
        eigenvalues = DiagonalMatrixOperator(input_shape, filter)
        super(FourierFilter, self).__init__(eigenvectors, eigenvalues)
        self.dim = dim
        self.filter = filter

class FourierConvolution(FourierFilter):
    def __init__(self, input_shape, kernel, dim):
        """
        This class implements a 2D Fourier convolution that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0), including for the input kernel

        it returns the Fourier transform with the zero-frequency component in the center of the image
        """
        filter = FourierTransform(input_shape, dim=dim).forward(kernel)
        super(FourierConvolution, self).__init__(input_shape, filter, dim)
