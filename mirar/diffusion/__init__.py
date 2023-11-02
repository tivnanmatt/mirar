import torch 
import os

from ..linalg.core import ScalarMatrixOperator, DiagonalMatrixOperator
from ..linalg.fourier import FourierMatrixOperator

class MatrixControlledDiffusionModel(torch.nn.Module):
    def __init__(self, H, Sigma, H_prime, Sigma_prime, score_estimator):       
        super(MatrixControlledDiffusionModel, self).__init__()
        self.H = H
        self.Sigma = Sigma
        self.H_prime = H_prime
        self.Sigma_prime = Sigma_prime
        self.score_estimator = score_estimator
        return

    def sample_x_t_given_x_0_and_t_and_white_noise( self, x_0, t, white_noise):
        
        # noise is assumed to be a tensor of random noise, normal distribution, mean=0, std=1

        # signal transfer matrix
        H_t = self.H(t)           
        
        # noise covariance matrix
        Sigma_t = self.Sigma(t) 
        
        # sample from p(x_t | x_0) = N(H_t * x_0, Sigma_t)  
        x_t = H_t * x_0 + Sigma_t.sqrt_MatrixOperator() * white_noise

        return x_t
    
    def sample_x_t_given_x_0_and_t( self, x_0, t):
        
        # sample random white noise, normal distribution, mean=0, std=1
        white_noise = torch.randn_like(x_0)

        # sample from p(x_t | x_0) = N(H_t * x_0, Sigma_t)
        x_t = self.sample_x_t_given_x_0_and_t_and_white_noise(x_0, t, white_noise)

        return x_t
    
    def sample_x_t_plus_tau_given_x_t_and_t_and_tau_and_white_noise(  self, x_t, t, tau, white_noise):
        
        # signal transfer matrix at time t
        H_t = self.H(t)

        # noise covariance matrix at time t
        Sigma_t = self.Sigma(t)

        # signal transfer matrix at time t+dt
        H_t_plus_tau = self.H(t+tau)

        # noise covariance matrix at time t+dt
        Sigma_t_plus_tau = self.Sigma(t+tau)

        # sample from p(x_t+tau | x_t) = N(M1 * x_t, M2)
        M1 = H_t_plus_tau @ H_t.inverse_MatrixOperator()
        M2 = Sigma_t_plus_tau -  M1 @ Sigma_t @ M1

        x_t_plus_tau = M1 * x_t + M2.sqrt() * white_noise

        return x_t_plus_tau
    
    def sample_x_t_plus_tau_given_x_t_and_t_and_tau(  self, x_t, t, tau):

        # sample random white noise, normal distribution, mean=0, std=1
        white_noise = torch.randn_like(x_t)

        # sample from p(x_t+tau | x_t) = N(M1 * x_t, M2)
        x_t_plus_tau = self.sample_x_t_plus_tau_given_x_t_and_t_and_tau_and_white_noise(x_t, t, tau, white_noise)

        return x_t_plus_tau
    
    def sample_x_t_plus_dt_given_x_t_and_t_and_dt(  self, x_t, t, dt):
        
        # this assumed dt is small, so we can use a first order approximation
        assert dt > 0, "dt must be positive"

        # signal transfer matrix at time t
        H_t = self.H(t)

        # noise covariance matrix at time t
        Sigma_t = self.Sigma(t)

        # time derivative of signal transfer matrix at time t
        H_prime_t = self.H_prime(t)

        # time derivative of noise covariance matrix at time t
        Sigma_prime_t = self.Sigma_prime(t)

        # define the coefficients of the stochastic differential equation
        F = H_prime_t @ H_t.inverse_MatrixOperator()
        f = F * x_t
        G2 = Sigma_prime_t - H_prime_t @ H_t.inverse_MatrixOperator() @ Sigma_t -  Sigma_t @ H_t.inverse_MatrixOperator() @ H_prime_t

        G = G2.sqrt_MatrixOperator()

        white_noise = torch.randn_like(x_t)

        # sample from p(x_t+dt | x_t) = N(x_t + f dt, G @ G dt)
        x_t_plus_dt = x_t + f*dt + G*torch.sqrt(dt)*white_noise

        return x_t_plus_dt
    
    def sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_score_estimator( self,
                                                                    x_t,
                                                                    t,
                                                                    dt,
                                                                    score_estimator,
                                                                    odeFlag=False,
                                                                    secondOrderFlag=False):
        
        # this assumed dt is small, so we can use a first order approximation
        assert dt > 0, "dt must be positive"

        # this method applies the Anderson formula for the reverse-time stochastic differential equation

        def compute_ode_gradient(x_t,t):

            # signal transfer matrix at time t
            H_t = self.H(t)

            # noise covariance matrix at time t
            Sigma_t = self.Sigma(t)

            # time derivative of signal transfer matrix at time t
            H_prime_t = self.H_prime(t)

            # time derivative of noise covariance matrix at time t
            Sigma_prime_t = self.Sigma_prime(t)

            # define the coefficients of the stochastic differential equation
            F = H_prime_t @ H_t.inv()
            f = F * x_t
            G2 = Sigma_prime_t - H_prime_t @ H_t.inv() @ Sigma_t -  Sigma_t @ H_t.inv() @ H_prime_t 

            G = G2.sqrt()
            
            score_estimate = score_estimator(x_t, t)

            # derivative of x_t with respect to t in the ODE
            return f - 0.5*G2*score_estimate, f, G, G2, score_estimate

        ode_gradient, f, G, G2, score_estimate = compute_ode_gradient(x_t, t)

        # sample from p(x_t-dt | x_t) = N(x_t - [f  - G @ G * score_estimate] dt, G @ G dt)
        
        if secondOrderFlag is False:

            if odeFlag:
                # first-order ODE sampler
                x_t_minus_dt = x_t - ode_gradient
            else:
                # first-order SDE sampler
                white_noise = torch.randn_like(x_t)
                x_t_minus_dt = x_t - ode_gradient - 0.5*f*dt + G2*score_estimate*dt + G*torch.sqrt(torch.tensor(dt))*white_noise

        else:

            k1 = ode_gradient
            k2, _, _, _, _ = compute_ode_gradient(x_t -dt*k1, t-dt)

            if odeFlag:
                # second-order ODE sampler
                x_t_minus_dt = x_t - 0.5*dt*(k1+k2)
            else:
                # second-order SDE sampler
                white_noise = torch.randn_like(x_t)
                x_t_minus_dt = x_t - 0.5*dt*(k1+k2) - 0.5*f*dt + G2*score_estimate*dt + G*torch.sqrt(torch.tensor(dt))*white_noise
        
        return x_t_minus_dt
    


class ScalarDiffusionModel(MatrixControlledDiffusionModel):
    def __init__(self,  signal_scale_func,
                        noise_variance_func,
                        signal_scale_derivative_func=None,
                        noise_variance_derivative_func=None,
                         **kwargs):
        
        
        def H(self, t):
            return ScalarMatrixOperator(self.signal_scale_func(t))
        
        if signal_scale_derivative_func is None:
            def H_prime(self, t):
                jacobians = [torch.autograd.functional.jacobian(self.signal_scale_func, t[i].unsqueeze(0)) for i in range(t.shape[0])]
                return ScalarMatrixOperator(torch.stack(jacobians).squeeze())
        else:
            def H_prime(self, t):
                return ScalarMatrixOperator(self.signal_scale_derivative_func(t))

        def Sigma(self, t):
            return ScalarMatrixOperator(self.noise_variance_func(t))
        
        if noise_variance_derivative_func is None:
            def Sigma_prime(self, t):
                jacobians = [torch.autograd.functional.jacobian(self.noise_variance_func, t[i].unsqueeze(0)) for i in range(t.shape[0])]
                return ScalarMatrixOperator(torch.stack(jacobians).squeeze())
        else:
            def Sigma_prime(self, t):
                return ScalarMatrixOperator(self.noise_variance_derivative_func(t))

        super(ScalarDiffusionModel, self).__init__(H, Sigma, H_prime, Sigma_prime, **kwargs)

        self.signal_scale_func = signal_scale_func
        self.noise_variance_func = noise_variance_func
        self.signal_scale_derivative_func = signal_scale_derivative_func
        self.noise_variance_derivative_func = noise_variance_derivative_func

        return

class VariancePreservingScalarDiffusionModel(ScalarDiffusionModel):
    def __init__(self,  alpha_bar_func,
                        alpha_bar_derivative_func=None,
                         **kwargs):
        
        # for the variance preserving model, the signal scale is defined as sqrt(alpha_bar_t)
        def signal_scale_func(t):
            return torch.sqrt(alpha_bar_func(t))

        if alpha_bar_derivative_func is None:
            signal_scale_derivative_func = None
        else:
            def signal_scale_derivative_func(t):
                return 0.5*torch.pow(alpha_bar_func(t), -0.5)*alpha_bar_derivative_func(t)

        # for the variance preserving model, the noise variance is defined as (1 - alpha_bar_t**2)
        def noise_variance_func(t):
            return 1.0 - alpha_bar_func(t)
        
        if alpha_bar_derivative_func is None:
            noise_variance_derivative_func = None
        else:
            def noise_variance_derivative_func(t):
                return -alpha_bar_derivative_func(t)

        super(VariancePreservingScalarDiffusionModel, self).__init__(   
                            signal_scale_func=signal_scale_func,
                            noise_variance_func=noise_variance_func,
                            signal_scale_derivative_func=signal_scale_derivative_func,
                            noise_variance_derivative_func=noise_variance_derivative_func,
                            **kwargs)
        return


class DiagonalDiffusionModel(MatrixControlledDiffusionModel):
    def __init__(self,  signal_scale_func,
                        noise_variance_func,
                        signal_scale_derivative_func=None,
                        noise_variance_derivative_func=None,
                         **kwargs):
        
        def H(self, t):
            return DiagonalMatrixOperator(self.signal_scale_func(t))
        
        if signal_scale_derivative_func is None:
            def H_prime(self, t):
                jacobians = [torch.autograd.functional.jacobian(self.signal_scale_func, t[i].unsqueeze(0)) for i in range(t.shape[0])]
                return DiagonalMatrixOperator(torch.stack(jacobians).squeeze())
        else:
            def H_prime(self, t):
                return DiagonalMatrixOperator(self.signal_scale_derivative_func(t))
            
        def Sigma(self, t):
            return DiagonalMatrixOperator(self.noise_variance_func(t))
        
        if noise_variance_derivative_func is None:
            def Sigma_prime(self, t):
                jacobians = [torch.autograd.functional.jacobian(self.noise_variance_func, t[i].unsqueeze(0)) for i in range(t.shape[0])]
                return DiagonalMatrixOperator(torch.stack(jacobians).squeeze())
        else:
            def Sigma_prime(self, t):
                return DiagonalMatrixOperator(self.noise_variance_derivative_func(t))
            
        super(DiagonalDiffusionModel, self).__init__(H, Sigma, H_prime, Sigma_prime, **kwargs)

        self.signal_scale_func = signal_scale_func
        self.noise_variance_func = noise_variance_func
        self.signal_scale_derivative_func = signal_scale_derivative_func
        self.noise_variance_derivative_func = noise_variance_derivative_func

        return


class FourierDiffusionModel(MatrixControlledDiffusionModel):
    def __init__(self,  
                 modulation_transfer_function_func,
                 noise_power_spectrum_func,
                 modulation_transfer_function_derivative_func=None,
                 noise_power_spectrum_derivative_func=None,
                 sample_x_T_func=None,
                 **kwargs):
        

        def H(self, t):
            return FourierMatrixOperator(self.modulation_transfer_function_func(t))
        
        if modulation_transfer_function_derivative_func is None:
            def H_prime(self, t):
                jacobians = [torch.autograd.functional.jacobian(self.modulation_transfer_function_func, t[i].unsqueeze(0)).squeeze(-1) for i in range(t.shape[0])]
                return FourierMatrixOperator(torch.stack(jacobians).squeeze(1))
        else:
            def H_prime(self, t):
                return FourierMatrixOperator(self.modulation_transfer_function_derivative_func(t))
            
        def Sigma(self, t):
            return FourierMatrixOperator(self.noise_power_spectrum_func(t))
        
        if noise_power_spectrum_derivative_func is None:
            def Sigma_prime(self, t):
                jacobians = [torch.autograd.functional.jacobian(self.noise_power_spectrum_func, t[i].unsqueeze(0)).squeeze(-1) for i in range(t.shape[0])]
                return FourierMatrixOperator(torch.stack(jacobians).squeeze(1))
        else:
            def Sigma_prime(self, t):
                return FourierMatrixOperator(self.noise_power_spectrum_derivative_func(t))
        
        super(FourierDiffusionModel, self).__init__(H, Sigma, H_prime, Sigma_prime, **kwargs)

        self.modulation_transfer_function_func = modulation_transfer_function_func
        self.noise_power_spectrum_func = noise_power_spectrum_func
        self.modulation_transfer_function_derivative_func = modulation_transfer_function_derivative_func
        self.noise_power_spectrum_derivative_func = noise_power_spectrum_derivative_func
        self.sample_x_T_func = sample_x_T_func

        return
