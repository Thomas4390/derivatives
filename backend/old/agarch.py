"""Further material on AGARCH, beyond the agarch class in garch.py"""
from .garch import *
from scipy.integrate import simps

"""
Bonjour à tous,

Voici les indices promis en classe hier. Désolé pour le retard; pour me faire pardonner, j'ai étendu le délai de remise de quelques jours. Je vous copie ci-bas le constructeur de ma classe agarch. J'ai aussi rebaptisé le notebook ngarch_estimation pour garch_estimation. Vous y remarquerez, en cellule [3], les lignes 5 & 6:

        warnings.warn('Assignment 2 hack')
        self.fix_parameter('omega', 0)        

C'est radical, il y a moyen de faire mieux, mais la valeur ajoutée de faire mieux ne serait pas très grande ici. Et comme certains articles publiés se le permettent, on se le permettra aussi.

Pour ce qui est des prix d'options, si votre code n'est pas assez rapide pour permettre de calculer les prix quotidiennement, faite le hebdomadairement (mercredi) ou mensuellement (dernier jour ouvrable).

Au plaisir!

Christian
"""
class HN2000(ModelParameters):
    """
    This is under the case where the affine kernel which obtains as xi = 0.
    """
    def __init__(self,phi_,DAYS=365):
        super().__init__()
        # Log grid
        self.phi_ = phi_
        self.alphaQ = 7.83e-7
        self.betaQ = 0.8810
        self.gammaQ = 378
        self.lambdaQ = 0
        self.h0 = 1.11e-4
        # Option contract example
        self.St = 100
        self.K = 100
        self.tau = 60
        self.rF = 0.05/DAYS
        self.htp1 = 0.21**2/DAYS
    

    def CharCoef(self, phi=None):
        if phi is None:
            phi = self.phi_
        aQ, bQ, gQ, lQ, h0, r, tau = self.alphaQ, self.betaQ, self.gammaQ, self.lambdaQ, self.h0, self.rF, self.tau
        
        # Infer omega (variance targeting):
        oQ = (1 - bQ - aQ*gQ**2) * h0 - aQ
        
        # Value to be used for integration
        ui = (phi - 1j)*1j
        
        A = np.zeros(shape=(tau,len(ui)), dtype=complex)
        B = np.zeros(shape=(tau,len(ui)), dtype=complex)
        
        # At time DTM-1
        A[0] = ui*r
        B[0] = -0.5*ui + 0.5*ui**2
        
        # Recursion backward in time
        for dtm in range(1,tau):
            A[dtm] = A[dtm-1] + ui*r + B[dtm-1]*oQ \
                                - 0.5*np.log(1 - 2*aQ*B[dtm-1])
            B[dtm] = ui*(gQ + lQ - 0.5) - 0.5*gQ**2 +bQ*B[dtm-1] \
                    + ( 0.5*(ui-gQ)**2 )/( 1-2*aQ*B[dtm-1] )
        
        return [A,B]
    
    def CharFunc(self, phi=None):
        if phi is None:
            phi = self.phi_
        St, tau, htp1 = self.St, self.tau, self.htp1
        
        A,B = self.CharCoef(phi)
        ui = (phi - 1j)*1j
        Psi = np.exp(np.log(St)*ui + A[tau-1] + B[tau-1]*htp1)
        
        return Psi
    
    def CPrice(self,K=None,phi=None):
        if phi is None:
            phi = self.phi_
        if K is None:
            K = self.K
            
        St,tau,r = self.St,self.tau, self.rF
        
        Psi = self.CharFunc(phi)
        integrand = np.imag( Psi*np.exp(-1j*phi*np.log(K) )/(1j*phi+1) )/phi
        CPrice = 0.5*St + np.exp( -r*tau )/np.pi * simps(integrand, phi)

        return CPrice
        
    def multiple_call(self,K,phi=None):
        """*Suboptimal* management of multiple strikes..."""
        if phi is None:
            phi = self.phi_
        
        c_prices = []
        for k in K:
            # Option prices for multiple strikes
            c_prices.append( self.CPrice(k,phi) )
        
        return c_prices
        
        
class HNPlot(HN2000):
    def __init__(self, phi_, *args, **kwargs):
        super().__init__(phi_, *args, **kwargs)
    
    def benchmark_call(self,K,N=20000, LB=np.log(1e-20),UB = np.log(400)):
        """
        Get the benchmark call values with multiple strikes
        """
        phi_dense = np.exp( np.linspace(LB, UB, N) )
        benchmark = self.multiple_call(K,phi_dense)
        
        return benchmark

    def benchmark_compare(self,K,N,LB,UB):
        """
        Compare the option prices with different log grids
        Attributes:
        N(array): The number of grid points
        LB(float): The lower boundary
        UB(float): The upper boundary
        """
        cgrid = np.zeros( shape=(len(K), len(N))  )
        dist  = np.zeros( shape=(len(K), len(N))  )
            
        for ni, n in enumerate(N):
            # Log grid
            phi = np.exp( np.linspace(LB, UB, n) )
                
            benchmark = self.benchmark_call(K)
            cgrid[:,ni] = self.multiple_call(K,phi)
            for ki in range(len(K)):
                # Evaluate option prices
                dist[ki,ni]  = 100*(cgrid[ki,ni] - benchmark[ki])/benchmark[ki]
        return cgrid,dist


class CGARCH(HN2000):
    def __init__(self,phi_,DAYS=365, BSvol=0.21, *args, **kwargs):
        super().__init__(phi_, *args, **kwargs)
        
        self.alphaQ = 1.808e-06
        self.betaQ = 9.297e-01
        self.gamma1Q = 5.854e+02
        self.gamma2Q = 5.749e+02
        self.omegaQ = 2.204e-07
        self.phiQ = 2.835e-07
        self.rhoQ = 9.966e-01
        # Two volatility components: SR and LR
        self.SR_tp1 = 0.1*BSvol**2/DAYS # h(t+1) - q(t+1), the 'short-run' component
        self.LR_tp1 = BSvol**2/DAYS   # q(t+1),          the 'long-run' component
        
    
    def CharCoef(self, phi=None):
        if phi is None:
            phi = self.phi_
        # ui = phi*1j
        
        # Value to be used for integration
        ui = (phi - 1j)*1j
        
        # Recover parameter values
        aQ, bQ, g1Q, g2Q, oQ, pQ, rQ, lQ, r, tau = self.alphaQ, self.betaQ,\
            self.gamma1Q, self.gamma2Q, self.omegaQ, self.phiQ, self.rhoQ, self.lambdaQ, self.rF, self.tau
        
        # Initialize weight matrices
        A  = np.zeros(shape=(tau, len(ui)), dtype=complex)
        B1 = np.zeros(shape=(tau, len(ui)), dtype=complex)
        B2 = np.zeros(shape=(tau, len(ui)), dtype=complex)
        
        # At DTM-1
        A[0]  = ui*r
        B1[0] = -0.5*ui + 0.5*ui**2
        B2[0] = -0.5*ui + 0.5*ui**2
        
        # Recursion backward in time
        for dtm in range(1,tau):
            A[dtm]  = A[dtm-1] + ui*r - (aQ*B1[dtm-1] + pQ*B2[dtm-1]) - \
                    0.5*np.log( 1-2*aQ*B1[dtm-1] -2*pQ*B2[dtm-1] ) + B2[dtm-1]*oQ
                    
            common  = 2*(aQ*g1Q*B1[dtm-1] + pQ*g2Q*B2[dtm-1] - 0.5*ui)**2 / \
                        (1 - 2*aQ*B1[dtm-1] - 2*pQ*B2[dtm-1])
                        
            B1[dtm] = bQ*B1[dtm-1] + (lQ - 0.5)*ui + common
            B2[dtm] = rQ*B2[dtm-1] + (lQ - 0.5)*ui + common
        
        return A, B1, B2
    
    def CharFunc(self, phi=None):
        if phi is None:
            phi = self.phi_
            
        St, tau, SR_tp1, LR_tp1 = self.St, self.tau, self.SR_tp1, self.LR_tp1
        A,B1,B2 = self.CharCoef(phi)
        # Value to be used for integration
        ui = (phi - 1j)*1j
        
        Psi = np.exp( np.log(St)*ui + A[tau-1] + B1[tau-1]*SR_tp1 + \
                    B2[tau-1]*LR_tp1  )
        
        return Psi
    
    def CPrice(self, K=None, phi=None):
        if phi is None:
            phi = self.phi_
        if K is None:
            K = self.K
               
        St,r,tau = self.St, self.rF, self.tau

        Psi = self.CharFunc(phi)
        integrand = np.imag( Psi*np.exp(-1j*phi*np.log(K) )/(1j*phi+1) )/phi
        CPrice    = 0.5*St + np.exp( -r*tau )/np.pi * simps(integrand, phi)
        
        return CPrice
    
    def multiple_call(self,K,phi=None):
        """*Suboptimal* management of multiple strikes..."""
        if phi is None:
            phi = self.phi_
        
        c_prices = []
        for k in K:
            # Option prices for multiple strikes
            c_prices.append( self.CPrice(k,phi) )
        
        return c_prices



class IGGARCH(HN2000):
    def __init__(self, phi_,DAYS=365, BSvol=0.21, *args, **kwargs):
        super().__init__(phi_, *args, **kwargs)
        self.muQ = 405.1
        self.omegaQ = 2.446e-16
        self.betaQ = -0.1902
        self.alphaQ = 49110
        self.cQ = 5.373e-06
        self.etaQ = -0.002465
        
        self.htp1 = BSvol ** 2 / DAYS
        
    def CharCoef(self, phi=None):
        if phi is None:
            phi = self.phi_
        mQ, oQ, bQ, aQ, cQ, eQ, tau, r = self.muQ, self.omegaQ,\
            self.betaQ, self.alphaQ, self.cQ, self.etaQ, self.tau, self.rF
        
        # Value to be used for integration
        ui = (phi - 1j)*1j
        
        # Initialize weight matrices
        A = np.zeros(shape=(tau, len(ui)), dtype=complex)
        B = np.zeros(shape=(tau, len(ui)), dtype=complex)
        
        # Few repetitive terms
        e2 = eQ ** 2
        e4 = eQ ** 4
        
        # At DTM-1
        A[0] = ui*r
        B[0] = mQ*ui + (1/e2) - (1/e2)*np.sqrt( 1-2*eQ*ui )
        
        # Recursion backward in time
        for dtm in range(1,tau):
            A[dtm] = A[dtm-1]    + ui*r  + B[dtm-1]*oQ - \
                    0.5*np.log( 1 - 2*aQ*e4*B[dtm-1] )
            B[dtm] = bQ*B[dtm-1] + mQ*ui + (1/e2) - \
                    (1/e2)*np.sqrt( (1-2*aQ*e4*B[dtm-1])* \
                                    (1-2*cQ*B[dtm-1] - 2*eQ*ui) )
        
        return A, B

    def CharFunc(self, phi=None):
        if phi is None:
            phi = self.phi_
        St, tau, htp1 = self.St, self.tau, self.htp1
        
        A,B = self.CharCoef(phi)
        # Value to be used for integration
        ui = (phi - 1j)*1j
        Psi = np.exp(np.log(St)*ui + A[tau-1] + B[tau-1]*htp1)
        
        return Psi
    
    def CPrice(self,K=None, phi=None):
        if phi is None:
            phi = self.phi_
        if K is None:
            K = self.K
        St, r, tau = self.St,self.rF, self.tau
        
        Psi = self.CharFunc(phi)
        integrand = np.imag( Psi*np.exp(-1j*phi*np.log(K) )/(1j*phi+1) )/phi
        CPrice    = 0.5*St + np.exp( -r*tau )/np.pi * simps(integrand, phi)
        
        return CPrice
    
    def multiple_call(self,K,phi=None):
        """*Suboptimal* management of multiple strikes..."""
        if phi is None:
            phi = self.phi_
        
        c_prices = []
        for k in K:
            # Option prices for multiple strikes
            c_prices.append( self.CPrice(k,phi) )
        
        return c_prices
        
     
     
# BCHJ, table 3 (MLE Options)
class CGARCH_IG(HN2000):
    def __init__(self, phi_,DAYS=365, BSvol=0.21, *args, **kwargs):
        super().__init__(phi_, *args, **kwargs)
        self.mu_tilde = -0.5 # Table gives us mu_tilde^*, but we'll need mu^*
        self.omegaqQ = 2.753e-07
        self.rho1Q = 0.985
        self.alphahQ = 3.054e+06
        self.chQ = 3.123e-06
        self.rho2Q = 0.9998
        self.alphaqQ = 3.426e+06
        self.cqQ = 2.101e-06
        self.etaQ = -8.630e-04
        
        self.qtp1 = BSvol ** 2 / DAYS
        self.htp1 = 0.9*self.qtp1
        
    def CharCoef(self, phi=None):
        if phi is None:
            phi = self.phi_
        mu_tilde, wqQ, r1Q, ahQ, chQ, r2Q, aqQ, cqQ, eQ, tau, r = self.mu_tilde, self.omegaqQ, \
            self.rho1Q, self.alphahQ, self.chQ, self.rho2Q, self.alphaqQ, self.cqQ, self.etaQ, self.tau, self.rF
        
        # Recover mu for CF (see equation 3(a))
        mQ  = mu_tilde - 1/eQ
            
        # Value to be used for integration
        ui = (phi - 1j)*1j

        # Initialize weight matrices
        A = np.zeros(shape=(tau, len(ui)), dtype=complex)
        B = np.zeros(shape=(tau, len(ui)), dtype=complex)
        C = np.zeros(shape=(tau, len(ui)), dtype=complex)
        
        # Few repetitive terms
        e2 = eQ ** 2
        e4 = eQ ** 4
        
        # At DTM-1
        A[0] = ui*r
        B[0] = mQ*ui + (1/e2) - (1/e2)*np.sqrt( 1-2*eQ*ui )
        C[0] = 0
        
        # Recursion backward in time
        for dtm in range(1,tau):
            A[dtm] = A[dtm-1] + ui*r  + (wqQ - ahQ*e4 - aqQ*e4)*B[dtm-1] + \
                    (wqQ - aqQ*e4)*C[dtm-1] - \
                    0.5*np.log( 1 - 2*(ahQ + aqQ)*e4*B[dtm-1] - 2*aqQ*e4*C[dtm-1] )
            
            F1 = 1 - 2*(aqQ + ahQ)*e4*B[dtm-1] - 2*aqQ*e4*C[dtm-1]
            F2 = 1 - 2*eQ*ui - 2*(cqQ + chQ)*B[dtm-1] - 2*cqQ*C[dtm-1]
            
            B[dtm] = mQ*ui + ( r1Q - (chQ + cqQ)/e2 -(ahQ + aqQ)*e2 )*B[dtm-1] - \
                    (cqQ/e2 + aqQ*e2)*C[dtm-1] + (1/e2) - \
                    np.sqrt( F1*F2 )/e2
                    
            C[dtm] = (r2Q - r1Q)*B[dtm-1] + r2Q*C[dtm-1]
        
        return A, B, C
    
    def CharFunc(self, phi=None):
        if phi is None:
            phi = self.phi_
        St, tau, htp1, qtp1 = self.St, self.tau, self.htp1, self.qtp1
        
        # Value to be used for integration
        ui = (phi - 1j)*1j
        
        A,B,C = self.CharCoef(phi)
        Psi = np.exp(np.log(St)*ui + A[tau-1] + B[tau-1]*htp1 + C[tau-1]*qtp1)
        
        return Psi
    
    def CPrice(self, K=None, phi=None):
        if phi is None:
            phi = self.phi_
        if K is None:
            K = self.K
        St, r, tau = self.St, self.rF, self.tau
        
        Psi = self.CharFunc(phi)
        integrand = np.imag( Psi*np.exp(-1j*phi*np.log(K) )/(1j*phi+1) )/phi
        CPrice    = 0.5*St + np.exp( -r*tau )/np.pi * simps(integrand, phi)
    
        return CPrice

    def multiple_call(self,K,phi=None):
        """*Suboptimal* management of multiple strikes..."""
        if phi is None:
            phi = self.phi_
        
        c_prices = []
        for k in K:
            # Option prices for multiple strikes
            c_prices.append( self.CPrice(k,phi) )
        
        return c_prices
        

