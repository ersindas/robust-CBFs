import numpy as np


class ControlAffine:
    """
    General Class for Nonlinear Control Affine System
    \dot{x} = f(x) + g(x)u
    """

    def __init__(self, f, g=lambda x: 1, n=1, m=1) -> None:
        """
        Inputs:
        f,g - functions specifying f(x) and g(x)
        n - state dimension
        m - input dimension
        """
        self.f = f
        self.g = g
        self.n = n
        self.m = m
        if m==1:
            self.RHS = self.single_RHS
        else:
            self.RHS = self.multi_RHS

    def single_RHS(self, x, u):
        return self.f(x) + self.g(x)*u
    
    def multi_RHS(self, x, u):
        return self.f(x) + self.g(x)@u


class LinearSystem(ControlAffine):
    """
    Special Case for
    \dot{x} = Ax + Bu
    """

    def __init__(self, A, B, C=0) -> None:
        if not isinstance(A, np.ndarray):
            super().__init__(lambda x: A*x, lambda x: B, 1)
        else:
            super().__init__(lambda x: A@x, lambda x: B, A.shape[0])
        self.C = C

    def output(self, x):
        return self.C*x


class DoubleIntegrator(LinearSystem):

    def __init__(self) -> None:
        super().__init__(np.array([[0, 1], [0, 0]]), np.array([0, 1]), np.array([1, 0]))