
import torch

class MatrixOperator(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        """
        This is an abstract class for linear operators.

        It inherits from torch.nn.Module.
        
        It requires the methods forward and adjoint to be implemented.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            output_shape: tuple of integers
                The shape of the output tensor, disregarding batch and channel dimensions.
        """

        super(MatrixOperator, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """

        return NotImplementedError
    
    def forward_MatrixOperator(self):
        return self
    
    def transpose(self,y):
        """
        This method returns the transpose of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the transpose of the linear operator.

        returns:
            transpose: MatrixOperator object
                The transpose of the linear operator.
        """
        
        # by default, use a forward pass applied to zeros and then transpose
        # this is not the most efficient way to do this, but it is the most general

        _input = torch.zeros(self.input_shape, dtype=torch.complex64, device=y.device)
        _input.requires_grad = True
        _output = self.forward(_input)
        # now use autograd to compute the transpose applied to y
        # we also need the transpose operation itself to be differentiable
        _output.backward(y, create_graph=True)
        return _input.grad
    
    def transpose_MatrixOperator(self):
        return TransposeMatrixOperator(self)
    
    def conjugate(self, x: torch.Tensor):
        """
        This method returns the conjugate of the linear operator.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the conjugate of the linear operator.

        returns:
            conjugate: MatrixOperator object
                The conjugate of the linear operator.

        """
        return torch.conj(self.forward(torch.conj(x)))

    def conjugate_MatrixOperator(self):
        return ConjugateMatrixOperator(self)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the adjoint pass of the linear operator, i.e. the conjugate-transposed matrix-vector product.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        return torch.conj(self.transpose(torch.conj(y)))

    def conjugate_transpose_MatrixOperator(self):
        return ConjugateTransposeMatrixOperator(self)    
    
    def _pseudo_inverse_weighted_average(self, x):
        """
        This method implements the pseudo inverse of the linear operator using a weighted average.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the conjugate_transpose of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the conjugate_transpose of the linear operator to the input tensor.
        """
        
        numerator = self.conjugate_transpose(x)
        
        denominator = self.conjugate_transpose(torch.ones_like(x))
        
        return numerator / (denominator + 1e-10)  # Avoid division by zero

    def _pseudo_inverse_conjugate_gradient(self, b, max_iter=1000, tol=1e-6, reg_strength=1e-3, verbose=False):
        """
        This method implements the pseudo inverse of the linear operator using the conjugate gradient method.

        It solves the linear system (A^T A + reg_strength * I) x = A^T b for x, where A is the linear operator.

        parameters:
            b: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to which the pseudo inverse of the linear operator should be applied.
            max_iter: int
                The maximum number of iterations to run the conjugate gradient method.
            tol: float
                The tolerance for the conjugate gradient method.
            reg_strength: float
                The regularization strength for the conjugate gradient method.
        returns:
            x_est: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the conjugate_transpose of the linear operator to the input tensor.
        """
        ATb = self.conjugate_transpose(b)
        x_est = self._pseudo_inverse_weighted_average(b)
        
        r = ATb - self.conjugate_transpose(self.forward(x_est)) - reg_strength * x_est
        p = r.clone()
        rsold = torch.dot(r.flatten(), r.flatten())
        
        for i in range(max_iter):
            if verbose:
                print("Inverting ", self.__class__.__name__, " with conjugate_gradient. Iteration: {}, Residual: {}".format(i, torch.sqrt(torch.abs(rsold))))
            ATAp = self.conjugate_transpose(self.forward(p)) + reg_strength * p
            alpha = rsold / torch.dot(p.flatten(), ATAp.flatten())
            x_est += alpha * p
            r -= alpha * ATAp
            rsnew = torch.dot(r.flatten(), r.flatten())
            if torch.sqrt(torch.abs(rsnew)) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            
        return x_est
    
    def pseudo_inverse(self, y, method=None, **kwargs):
        """
        This method implements the pseudo inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to which the pseudo inverse of the linear operator should be applied.
            method: str
                The method to use for computing the pseudo inverse. If None, the method is chosen automatically.
            kwargs: dict
                Keyword arguments to be passed to the method.
        """

        if method is None:
            method = "conjugate_gradient"

        assert method in ["weighted_average", "conjugate_gradient"], "The method should be either 'weighted_average' or 'conjugate_gradient'."

        if method == "weighted_average":
            result =  self._pseudo_inverse_weighted_average(y, **kwargs)
        elif method == "conjugate_gradient":
            result =  self._pseudo_inverse_conjugate_gradient(y, **kwargs)

        return result
    
    def mat_add(self, M):
        raise NotImplementedError
    def mat_sub(self, M):
        raise NotImplementedError
    def mat_mul(self, M):
        raise NotImplementedError
    def __mul__(self, x):
        return self.forward(x)
    def __add__(self, M):
        return self.mat_add(M)
    def __sub__(self, M):
        return self.mat_sub(M)
    def __matmul__(self, M):
        return self.mat_mul(M)

    
class RealMatrixOperator(MatrixOperator):
    def __init__(self, input_shape, output_shape):
        """
        This is an abstract class for real linear operators.

        It inherits from MatrixOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            output_shape: tuple of integers
                The shape of the output tensor, disregarding batch and channel dimensions.
        """

        super(RealMatrixOperator, self).__init__(input_shape, output_shape)

    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the conjugate of the linear operator.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the conjugate of the linear operator.

        returns:
            conjugate: MatrixOperator object
                The conjugate of the linear operator.

        """
        self.forward(x)        
    
    def conjugate_MatrixOperator(self):
        return self
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.transpose(y)
    
    def conjugate_transpose_MatrixOperator(self):
        return TransposeMatrixOperator(self)
    

    

class SquareMatrixOperator(MatrixOperator):
    def __init__(self, input_shape):
        """
        This is an abstract class for square linear operators.

        It inherits from MatrixOperator.

        For square linear operators, the input and output shapes are the same.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        super(SquareMatrixOperator, self).__init__(input_shape, input_shape)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of appl\ng the inverse linear operator to the input tensor.
        """
        raise self.inverse_MatrixOperator().forward(y)
    
    def inverse_MatrixOperator(self):
        """
        This method returns the inverse of the linear operator.

        returns:
            inverse: MatrixOperator object
                The inverse of the linear operator.
        """
        raise NotImplementedError
        
class UnitaryMatrixOperator(MatrixOperator):
    def __init__(self, input_shape):
        """
        This is an abstract class for unitary linear operators.

        It inherits from InvertibleMatrixOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        super(UnitaryMatrixOperator, self).__init__(input_shape, input_shape)

    def inverse_MatrixOperator(self):
        return self.conjugate_transpose_MatrixOperator()


class HermitianMatrixOperator(SquareMatrixOperator):
    def __init__(self, input_shape):
        """
        This is an abstract class for Hermitian, or self-adjoint linear operators.

        It inherits from SquareMatrixOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        super(HermitianMatrixOperator, self).__init__(input_shape)

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.forward(y)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.transpose(x)
    

class SymmetricMatrixOperator(SquareMatrixOperator):
    def __init__(self, input_shape):
        """
        This is an abstract class for Symmetric linear operators.

        It inherits from SquareMatrixOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        super(SymmetricMatrixOperator, self).__init__(input_shape)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the transpose of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the transpose of the linear operator to the input tensor.
        """
        return self.forward(y)
    
    def cojugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the adjoint pass of the linear operator, i.e. the matrix-vector product with the adjoint.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """
        return self.conjugate(y)

class ScalarMatrixOperator(SymmetricMatrixOperator):
    def __init__(self, input_shape, scalar):
        """
        This class implements a scalar linear operator.

        It inherits from SymmetricMatrixOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            scalar: float
                The scalar to multiply the input tensor with.
        """

        super(ScalarMatrixOperator, self).__init__(input_shape)

        self.scalar = scalar

    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """

        return self.scalar * x
    
    def conjugate(self, x):
        """
        This method implements the adjoint pass of the linear operator, i.e. the conjugate-transposed matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """
        return torch.conj(self.scalar) * x
    
    def inverse_MatrixOperator(self):
        if self.scalar == 0:
            raise ValueError("The scalar is zero, so the inverse does not exist.")
        return ScalarMatrixOperator(self.input_shape, 1/self.scalar)
    
    def mat_add(self, added_scalar_matrix):
        assert isinstance(added_scalar_matrix, (ScalarMatrixOperator)), "ScalarMatrixOperator addition only supported for ScalarMatrixOperator." 
        scalar = ScalarMatrixOperator(self.input_shape, self.scalar + added_scalar_matrix.scalar)
    
    def mat_sub(self, sub_scalar_matrix):
        assert isinstance(sub_scalar_matrix, (ScalarMatrixOperator)), "ScalarMatrixOperator subtraction only supported for ScalarMatrixOperator." 
        scalar = ScalarMatrixOperator(self.input_shape, self.scalar - sub_scalar_matrix.scalar)

    def mat_mul(self, mul_scalar_matrix):
        assert isinstance(mul_scalar_matrix, (ScalarMatrixOperator)), "ScalarMatrixOperator multiplication only supported for ScalarMatrixOperator." 
        scalar = ScalarMatrixOperator(self.input_shape, self.scalar * mul_scalar_matrix.scalar)


class DiagonalMatrixOperator(SquareMatrixOperator):
    def __init__(self, input_shape, diagonal_vector):
        """
        This class implements a diagonal linear operator.

        It inherits from SquareMatrixOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            diagonal_vector: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The diagonal of the linear operator.
        """

        super(DiagonalMatrixOperator, self).__init__(input_shape)

        self.diagonal_vector = diagonal_vector
    
    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """
        return self.diagonal_vector * x
    
    def conjugate(self, x):
        return torch.conj(self.diagonal_vector) * x
    
    def inverse_MatrixOperator(self):
        if torch.any(self.diagonal_vector == 0):
            raise ValueError("The diagonal vector contains zeros, so the inverse does not exist.")
        return DiagonalMatrixOperator(self.input_shape, 1/self.diagonal_vector)
    
    def mat_add(self, added_diagonal_matrix):
        assert isinstance(added_diagonal_matrix, (DiagonalMatrixOperator)), "DiagonalMatrixOperator addition only supported for DiagonalMatrixOperator." 
        assert self.input_shape == added_diagonal_matrix.input_shape, "DiagonalMatrixOperator addition only supported for DiagonalMatrixOperator with same input shape."
        return DiagonalMatrixOperator(self.input_shape, self.diagonal_vector + added_diagonal_matrix.diagonal_vector)

    def mat_sub(self, sub_diagonal_matrix):
        assert isinstance(sub_diagonal_matrix, (DiagonalMatrixOperator)), "DiagonalMatrixOperator subtraction only supported for DiagonalMatrixOperator." 
        assert self.input_shape == sub_diagonal_matrix.input_shape, "DiagonalMatrixOperator subtraction only supported for DiagonalMatrixOperator with same input shape."
        return DiagonalMatrixOperator(self.input_shape, self.diagonal_vector - sub_diagonal_matrix.diagonal_vector)
    
    def mat_mul(self, mul_diagonal_matrix):
        assert isinstance(mul_diagonal_matrix, (DiagonalMatrixOperator)), "DiagonalMatrixOperator multiplication only supported for DiagonalMatrixOperator." 
        assert self.input_shape == mul_diagonal_matrix.input_shape, "DiagonalMatrixOperator multiplication only supported for DiagonalMatrixOperator with same input shape."
        return DiagonalMatrixOperator(self.input_shape, self.diagonal_vector * mul_diagonal_matrix.diagonal_vector)


class IdentityMatrixOperator(RealMatrixOperator, UnitaryMatrixOperator, HermitianMatrixOperator, SymmetricMatrixOperator):
    def __init__(self, input_shape):
        """
        This class implements the identity linear operator.

        It inherits from SquareMatrixOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        SquareMatrixOperator.__init__(self, input_shape)

    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """
        return x
    



class ConjugateMatrixOperator(MatrixOperator):
    def __init__(self, base_matrix_operator: MatrixOperator):
        """
        This is an abstract class for linear operators that are the conjugate of another linear operator.

        It inherits from MatrixOperator.

        parameters:
            base_matrix_operator: MatrixOperator object
                The linear operator to which the conjugate should be applied.
        """
            
        assert isinstance(base_matrix_operator, MatrixOperator), "The linear operator should be a MatrixOperator object."
        super(ConjugateMatrixOperator, self).__init__(base_matrix_operator.output_shape, base_matrix_operator.input_shape)

        self.base_matrix_operator = base_matrix_operator  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(y)
    
class TransposeMatrixOperator(MatrixOperator):
    def __init__(self, base_matrix_operator: MatrixOperator):
        """
        This is an abstract class for linear operators that are the transpose of another linear operator.

        It inherits from MatrixOperator.

        parameters:
            base_matrix_operator: MatrixOperator object
                The linear operator to which the conjugate should be applied.
        """
            
        assert isinstance(base_matrix_operator, MatrixOperator), "The linear operator should be a MatrixOperator object."

        super(TransposeMatrixOperator, self).__init__(base_matrix_operator.output_shape, base_matrix_operator.input_shape)

        self.base_matrix_operator = base_matrix_operator  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(y)

class ConjugateTransposeMatrixOperator(MatrixOperator):
    def __init__(self, base_matrix_operator: MatrixOperator):
        """
        This is an abstract class for linear operators that are the conjugate transpose of another linear operator.

        It inherits from MatrixOperator.

        parameters:
            base_matrix_operator: MatrixOperator object
                The linear operator to which the conjugate should be applied.
        """
            
        assert isinstance(base_matrix_operator, MatrixOperator), "The linear operator should be a MatrixOperator object."

        super(ConjugateTransposeMatrixOperator, self).__init__(base_matrix_operator.output_shape, base_matrix_operator.input_shape)

        self.base_matrix_operator = base_matrix_operator
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(y)



class CompositeMatrixOperator(MatrixOperator):
    def __init__(self, matrix_operators):
        """
        This class represents the matrix-matrix product of multiple linear operators.

        It inherits from MatrixOperator.

        parameters:
            operators: list of MatrixOperator objects
                The list of linear operators to be composed. The product is taken in the order they are provided.
        """

        assert isinstance(matrix_operators, list), "The operators should be provided as a list of MatrixOperator objects."
        assert len(matrix_operators) > 0, "At least one operator should be provided."
        for operator in matrix_operators:
            assert isinstance(operator, MatrixOperator), "All operators should be MatrixOperator objects."

        # The input shape of the composite operator is the input shape of the first operator,
        # and the output shape is the output shape of the last operator.
        input_shape = matrix_operators[0].input_shape
        output_shape = matrix_operators[-1].output_shape

        MatrixOperator.__init__(self, input_shape, output_shape)

        self.matrix_operators = matrix_operators

    def forward(self, x):
        result = x
        for matrix_operator in self.matrix_operators:
            result = matrix_operator.forward(result)
        return result
    
    def conjugate(self, x):
        result = x
        for matrix_operator in self.matrix_operators:
            result = matrix_operator.conjugate(result)
        return result
    
    def transpose(self,y):
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.transpose(result)
        return result
    
    def inverse_MatrixOperator(self):
        return CompositeMatrixOperator([matrix_operator.inverse_MatrixOperator() for matrix_operator in reversed(self.matrix_operators)])
    
    def to (self, x):
        for matrix_operator in self.matrix_operators:
            matrix_operator.to(x)
        return self




class EigenDecomposedMatrixOperator(CompositeMatrixOperator):
    def __init__(self, input_shape, eigenvalues: DiagonalMatrixOperator, eigenvectors: UnitaryMatrixOperator):
        """
        This class represents a linear operator that is given by its eigenvalue decomposition.

        It inherits from SquareMatrixOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            eigenvalues: DiagonalMatrixOperator object
                The diagonal matrix of eigenvalues.
            eigenvectors: UnitaryMatrixOperator object
                The matrix of eigenvectors.
        """

        assert isinstance(eigenvalues, DiagonalMatrixOperator), "The eigenvalues should be a DiagonalMatrixOperator object."
        assert isinstance(eigenvectors, UnitaryMatrixOperator), "The eigenvectors should be a UnitaryMatrixOperator object."
        assert eigenvalues.input_shape == input_shape, "The input shape of the eigenvalues should be the same as the input shape of the linear operator."
        assert eigenvectors.input_shape == input_shape, "The input shape of the eigenvectors should be the same as the input shape of the linear operator."
        assert eigenvalues.output_shape == input_shape, "The output shape of the eigenvalues should be the same as the input shape of the linear operator."
        assert eigenvectors.output_shape == input_shape, "The output shape of the eigenvectors should be the same as the input shape of the linear operator."

        operators = [eigenvectors, eigenvalues, eigenvectors.inverse_MatrixOperator()]

        super(EigenDecomposedMatrixOperator, self).__init__(operators)

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

class SingularValueDecomposedMatrixOperator(CompositeMatrixOperator):
    def __init__(self, input_shape, left_singular_vectors: UnitaryMatrixOperator, singular_values: DiagonalMatrixOperator,  right_singular_vectors: UnitaryMatrixOperator):
        """
        This class represents a linear operator that is given by its singular value decomposition.

        It inherits from SquareMatrixOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            singular_values: DiagonalMatrixOperator object
                The diagonal matrix of singular values.
            left_singular_vectors: UnitaryMatrixOperator object
                The matrix of left singular vectors.
            right_singular_vectors: UnitaryMatrixOperator object
                The matrix of right singular vectors.
        """

        assert isinstance(singular_values, DiagonalMatrixOperator), "The singular values should be a DiagonalMatrixOperator object."
        assert isinstance(left_singular_vectors, UnitaryMatrixOperator), "The left singular vectors should be a UnitaryMatrixOperator object."
        assert isinstance(right_singular_vectors, UnitaryMatrixOperator), "The right singular vectors should be a UnitaryMatrixOperator object."
        assert singular_values.input_shape == input_shape, "The input shape of the singular values should be the same as the input shape of the linear operator."
        assert left_singular_vectors.input_shape == input_shape, "The input shape of the left singular vectors should be the same as the input shape of the linear operator."
        assert right_singular_vectors.input_shape == input_shape, "The input shape of the right singular vectors should be the same as the input shape of the linear operator."
        assert singular_values.output_shape == input_shape, "The output shape of the singular values should be the same as the input shape of the linear operator."
        assert left_singular_vectors.output_shape == input_shape, "The output shape of the left singular vectors should be the same as the input shape of the linear operator."
        assert right_singular_vectors.output_shape == input_shape, "The output shape of the right singular vectors should be the same as the input shape of the linear operator."

        operators = [left_singular_vectors, singular_values, right_singular_vectors.conjugate_transpose_MatrixOperator()]

        super(SingularValueDecomposedMatrixOperator, self).__init__(operators)

        self.singular_values = singular_values
        self.left_singular_vectors = left_singular_vectors
        self.right_singular_vectors = right_singular_vectors

