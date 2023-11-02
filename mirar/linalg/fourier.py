
import torch

from .core import UnitaryMatrixOperator, DiagonalMatrixOperator, EigenDecomposedMatrixOperator
    
class FourierTransform(UnitaryMatrixOperator):
    def __init__(self, input_shape, dim):
        """
        This class implements a N-Dimensional Fourier transform that can be used in a PyTorch model.

        it assumes the central pixel in the image is at the center of the input tensor in all dimensions

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

class FourierMatrixOperator(EigenDecomposedMatrixOperator):
    def __init__(self, input_shape, filter, dim):
        """
        This class implementts a ND Fourier filter that can be used in a PyTorch model.

        it assumes the central pixel in the image is at the center of the input tensor in all dimensions

        it returns the Fourier filter applied to the input. 
        """
        eigenvectors = FourierTransform(input_shape, dim=dim)
        eigenvalues = DiagonalMatrixOperator(input_shape, filter)
        super(FourierMatrixOperator, self).__init__(eigenvectors, eigenvalues)
        self.dim = dim
        self.filter = filter

    def mat_add(self, added_fourier_filter):
        assert isinstance(added_fourier_filter, (FourierMatrixOperator)), "FourierMatrixOperator addition only supported for FourierMatrixOperator." 
        assert self.input_shape == added_fourier_filter.input_shape, "FourierMatrixOperator addition only supported for FourierMatrixOperator with same input shape."
        assert self.dim == added_fourier_filter.dim, "FourierMatrixOperator addition only supported for FourierMatrixOperator with same dim."
        return FourierMatrixOperator(self.input_shape, self.filter + added_fourier_filter.filter, dim=self.dim)
    
    def mat_sub(self, sub_fourier_filter):
        assert isinstance(sub_fourier_filter, (FourierMatrixOperator)), "FourierMatrixOperator subtraction only supported for FourierMatrixOperator."
        assert self.input_shape == sub_fourier_filter.input_shape, "FourierMatrixOperator subtraction only supported for FourierMatrixOperator with same input shape."
        assert self.dim == sub_fourier_filter.dim, "FourierMatrixOperator subtraction only supported for FourierMatrixOperator with same dim."
        return FourierMatrixOperator(self.input_shape, self.filter - sub_fourier_filter.filter, dim=self.dim)
    
    def mat_mul(self, mul_fourier_filter):
        assert isinstance(mul_fourier_filter, (FourierMatrixOperator)), "FourierMatrixOperator multiplication only supported for FourierMatrixOperator."
        assert self.input_shape == mul_fourier_filter.input_shape, "FourierMatrixOperator multiplication only supported for FourierMatrixOperator with same input shape."
        assert self.dim == mul_fourier_filter.dim, "FourierMatrixOperator multiplication only supported for FourierMatrixOperator with same dim."
        return FourierMatrixOperator(self.input_shape, self.filter * mul_fourier_filter.filter, dim=self.dim)
    

class FourierConvolution(FourierMatrixOperator):
    def __init__(self, input_shape, kernel, dim):
        """
        This class implements a 2D Fourier convolution that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0), including for the input kernel

        it returns the Fourier transform with the zero-frequency component in the center of the image
        """
        filter = FourierTransform(input_shape, dim=dim).forward(kernel)
        super(FourierConvolution, self).__init__(input_shape, filter, dim)
