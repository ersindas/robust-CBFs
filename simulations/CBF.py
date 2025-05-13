import numpy as np
import osqp
from scipy import sparse


class Tun_Rob_CBF:
    """
    Tunable Robust CBF Function class
    """

    def __init__(self, a, h, dh, sys, k1, k2, eps):
        """"
        Inputs:
        a - value of constant class K function in CBF inequality
        h - CBF
        dh - gradient vector of h w.r.t x
        sys - ControlAffine system object
        k1, k2 - constants in R-CBF
        eps - epsilon function in Tunable case
        """
        self.a = a
        self.h = h
        self.dh = dh
        self.sys = sys
        if sys.n==1:
            self.Lfh = lambda x: dh(x)*sys.f(x)
            self.Lgh = lambda x: dh(x)*sys.g(x)
        else:
            self.Lfh = lambda x: dh(x)@sys.f(x)
            self.Lgh = lambda x: dh(x)@sys.g(x)
        self.k1 = k1
        self.k2 = k2
        self.eps = eps
        self.thresh = 0
        if sys.m==1:
            self.filter = self.filter_single
        else:
            self.filter = self.filter_multi

    def filter_single(self, x, u):
        self.thresh = ((self.k1/self.eps(self.h(x)))*np.abs(self.Lgh(x))+(self.k2/self.eps(self.h(x)))*self.Lgh(x)*self.Lgh(x)-self.Lfh(x)-self.a*self.h(x))/self.Lgh(x)
        if self.Lgh(x)>0:
            return max(u, self.thresh)
        if self.Lgh(x)<0:
            return min(u, self.thresh)
        return u
    
    def filter_multi(self, x, u):
        val = self.Lfh(x)+self.a*self.h(x)-(self.k1/self.eps(self.h(x)))*np.linalg.norm(self.Lgh(x))-(self.k2/self.eps(self.h(x)))*(self.Lgh(x)@self.Lgh(x))
        if self.Lgh(x)@u+val >= 0:
            return u
        else:
            return -self.Lgh(x)*val/(self.Lgh(x)@self.Lgh(x))
    

class Rob_CBF(Tun_Rob_CBF):
    """
    Non Tunable Version
    """

    def __init__(self, a, h, dh, sys, k1, k2):
        super().__init__(a, h, dh, sys, k1, k2, lambda y:1)


class CBF(Rob_CBF):
    """
    Regular CBF
    """

    def __init__(self, a, h, dh, sys):
        super().__init__(a, h, dh, sys, 0, 0)
        

class MR_CBF(CBF):
    """
    MR-CBF (for SOCP)
    """

    def __init__(self, a, h, dh, sys, d, Lip_Lfh, Lip_Lgh, Lip_ah):
        super().__init__(a, h, dh, sys)
        self.Lip_Lfh = Lip_Lfh
        self.Lip_Lgh = Lip_Lgh
        self.Lip_ah = Lip_ah
        self.d=d

    def filter(self, x, u):
        if np.abs(x)>1e6:
            return 0
        x = x[0]
        fx = self.d*(self.Lip_Lfh(x, self.d)+self.Lip_ah(x, self.a, self.d)) - self.Lfh(x) - self.a*self.h(x)
        
        P = sparse.csc_matrix([[1]])
        q = np.array([-u])
        A = sparse.csc_matrix([[self.Lgh(x)-self.d*self.Lip_Lgh(x, self.d)],[self.Lgh(x)+self.d*self.Lip_Lgh(x,self.d)]])
        l = np.array([fx, fx])
        u = np.array([np.inf, np.inf])
        
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, alpha=1.0, verbose=False)
        res = prob.solve()

        # if res.info.status != 'solved':
        #     print(x)
        #     print(A.toarray(), l, u)
        #     print(f"Solver failed with status: {res.info.status}")
        if res.x[0] is None:
            return 0
        return res.x[0]