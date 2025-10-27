"""A test suite on the outputs of sp_examples.ipynb

Create a test suite with 1 test per product in the notebook.

Start by the first product and contact me as soon as it is done. 
The test function should accept a boolean "update": when True, the test is not ran as a test per se, but saves a pickle with the testable output. 
This pkl will be loaded as expected results when update is false.
"""
import inspect 
import unittest
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .instruments import Bond, Spot, Call, Put, DownAndOutPut, DigitalCall, DigitalPut

"""
to launch the program in the cmd : python test_nbc_notes.py
"""
class TestStructuredProducts(unittest.TestCase):
    update = False
    class Register:
        def __init__(self, maturity=None, barriers=None, acceleration=None):
            self.maturity = maturity
            self.barriers = barriers
            self.acceleration = acceleration
            
    registered = Register(maturity=[], barriers=[], acceleration=[])

    
    @classmethod
    def register(cls, T, barriers=None, acceleration=None):
        cls.registered.maturity.append(T)
        if barriers is not None:
            cls.registered.barriers += barriers
        if acceleration is not None:
            cls.registered.acceleration.append(acceleration)
    
    def setUp(self):
        """Preparing the necessary setup for each test."""
        self.S_0 = 100
        self.K = self.S_0 # for all products K=S_0

        # Defining these here so that all graphs are anchored around the same level
        self.S_T = np.arange(0.25, 1.65, 0.01)*self.S_0        

    def initialize_test(self, update):
        # Use the calling function name as the test name
        test_name = inspect.stack()[1][3]
        if update is None:
            update = self.update

        path_to_expected_results = os.path.join('data','test','sp','%s.pkl'%test_name.replace('test_',''))
        if not update:
            assert os.path.exists(path_to_expected_results), \
                "Generate expected results before running test %s"%test_name

        return path_to_expected_results, update

    def finalize_test(self, results, filename, update):
        if update: 
            with open(filename, 'wb') as fh: 
                pickle.dump(results, fh)
            return

        # Compare results with the expected results
        with open(filename, 'rb') as fh: 
            expected = pickle.load(fh)
            self.assertTrue(np.array_equal(expected, results))

    def test_nbc1155(self, update=None):
        """NBC1155: Recovery Note Securities linked to the S&P/TSX 60 Index Series 4         
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=600

        Down and out put option ($K=100, H=65$) + underlying asset (S&P/TSX 60 Index) 

        Path dependence:
        * If the put is alive: 

        | Terminal value   | Payoff |
        |------------------|--------|
        | H < S(T) $\le$ K | K      |
        | K < S(T).        | S      |
        
        * If knocked out: Payoff = S(T)       
        """
        filename, update = self.initialize_test(update)
        
        T = 3
        H = 0.65*self.S_0
        self.register(T, barriers=[H])
        
        # A structured producted is often referred to as a "note"
        self.note = DownAndOutPut(self.S_0, self.K, H) + Spot(self.S_0)

        # The down and out put is path dependent. These "paths" have an initial row at S_0 and, for simplicity an
        # "intermediate" time step, also at S_0 > H.
        S_0_t = np.array([self.S_0, self.S_0]).reshape(-1,1)
        self.note.update_history(S_0_t)
        
        # This alternate set of "paths" all cross H at the intermediate date 
        self.note_out = DownAndOutPut(self.S_0, self.K, H) + Spot(self.S_0)
        S_0_t = np.array([self.S_0, H-1]).reshape(-1,1)

        self.note_out.update_history(S_0_t)
        results=np.concatenate(( self.note_out.payoff(self.S_T),self.note.payoff(self.S_T) ))

        self.finalize_test(results, filename, update)
        
        
    def test_v75863(self,update=None):
        """V75863, NBC Recovery Note Securities linked to the S&P/TSX 60 Index S1
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=625

        Down and out put option ($K=100, L=65$) 
         + underlying asset (S&P/TSX 60 Index) 
         - call option with strike $H=150$
        
        Path dependence:
         * If the put is alive: 

         | Terminal value   | Payoff |
         |------------------|--------|
         | L < S(T) $\le$ K | K      |
         | K < S(T) < H     | S      |
         | H $\le$ S(T)     | H      |
         
        * If knocked out:

         | Terminal value   | Payoff |
         |------------------|--------|
         | S(T) < H         | S      |
         | H $\le$ S(T)     | H      |
        """
        filename, update = self.initialize_test(update)
        
        T = 2
        L = 0.6*self.S_0
        H = 1.5*self.S_0
        self.register(T, barriers=[L,H])
        
        # A structured producted is often referred to as a "note"
        self.note = DownAndOutPut(self.S_0, self.K, L) + Spot(self.S_0) - Call(self.S_0, H)
        
        # The down and out put is path dependent. These "paths" have an initial row at S_0 and, for simplicity an
        # "intermediate" time step, also at S_0 > L.
        S_0_t = np.array([self.S_0, self.S_0]).reshape(-1,1)
        self.note.update_history(S_0_t)
        
        # This alternate set of "paths" all cross L at the intermediate date 
        self.note_out = DownAndOutPut(self.S_0, self.K, L) + Spot(self.S_0)
        S_0_t = np.array([self.S_0,L-1]).reshape(-1,1)
        self.note_out.update_history(S_0_t)
        
        # This alternate set of "paths" all cross H at the intermediate date 
        self.note_out = DownAndOutPut(self.S_0, self.K, H) + Spot(self.S_0)
        S_0_t = np.array([self.S_0, H-1]).reshape(-1,1)
        self.note_out.update_history(S_0_t)
        
        results = np.concatenate(( self.note_out.payoff(self.S_T), self.note.payoff(self.S_T) ))
        
        self.finalize_test(results, filename, update)
       
    def test_nbc1280(self,update=None): 
        """NBC1280, NBC Marathon Note Securities (Maturity-Monitored Barrier) linked to the units of the iShares® S&P/TSX 60
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1167

        * Payoff table: 

        | Terminal value    | Payoff |
        |-------------------|--------|
        | S(T) $\le$ H      | S      |
        | H< S(T) $\le$ S_0 | S_0    |
        | S(T)>S_0.         | AF*S   |        
        """
        filename, update = self.initialize_test(update)
        
        T = 5
        H = 0.75*self.S_0
        AF = 1.625 # Acceleration Factor
        self.register(T, barriers=[H], acceleration=AF)

        # A structured producted is often referred to as a "note"
        self.note_spot_put = Spot(self.S_0) + Put(self.S_0, self.K) - Put(self.S_0, H) 
        self.note_digitalput =- (self.K-H)*DigitalPut(np.nan, H) + (AF-1)*Call(self.S_0, self.K)
        self.note =self.note_spot_put  +self.note_digitalput 

        results = self.note.payoff(self.S_T)

        self.finalize_test(results, filename, update)
        

    def test_nbc1282(self,update=None):  
        """NBC1282, NBC Marathon Note Securities (Partial Protection) linked to the units of the iShares® S&P/TSX 60 Index ETF        
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1170
        
        * Payoff table: 
            
        | Terminal value   | Payoff           |
        |------------------|------------------|
        | S(T) $\le$ H     | H*S_0            |
        | H < S(T) $\le$ K | S                |
        | S(T)>K.          | S+(AF-1)*(S-S_0) |
        """
        filename, update = self.initialize_test(update)  
        
        T = 5
        H = 0.85*self.S_0
        AF = 1.2 # Acceleration Factor
        self.register(T, barriers=[H], acceleration=AF)
        
        # A structured producted is often referred to as a "note"
        self.note = Spot(self.S_0) + Put(self.S_0, H) + (AF-1)*Call(self.S_0, self.K)

        results = self.note.payoff(self.S_T)

        self.finalize_test(results, filename, update)
        
    def test_nbc2709(self,update=None):
        """NBC2709, NBC Bonus TwoStep Note Securities (Maturity-Monitored Barrier) linked to the units of the iShares® S&P/TSX 60 Index ETF

        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1188


        * Payoff table: 
        | Terminal value                 | Payoff                                       |
        |--------------------------------|----------------------------------------------|
        | S(T) $\le$ H                   | S                                            |
        | H < S(T) $\le$ S_0             | S_0                                          |
        | S_0 < S(T) $\le$ S_0*(1+bonus) | S_0*(1+bonus)                                |
        | S(T)>S_0*(1+bonus).            | S_0*(1+bonus+AF*((S-S_0)/S_0-Bonus))=S, AF=1 |
    
        """
        
        
        filename, update = self.initialize_test(update)  
        T = 5.5
        H = 0.75*self.S_0
        
        Bonus = 0.3625
        K_1 = self.S_0*(1+Bonus)
        self.register(T, barriers=[H])
        
        self.note_digitalput = self.K*DigitalPut(np.nan, self.K) - Put(self.S_0, H) - (self.K-H)*DigitalPut(np.nan,H) 
        self.note_digitalcall =K_1*DigitalCall(np.nan, self.K) + Call(self.S_0, K_1)
        self.note = self.note_digitalput + self.note_digitalcall 
        results=self.note.payoff(self.S_T)
        
        self.finalize_test(results, filename, update)
        
        
    
    def test_nbc1069(self,update=None):
        """NBC1069, NBC Trekking Note Securities (Twin Win) linked to the units of the iShares® S&P/TSX 60 Index ETF

        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1189


        * Payoff table: 

        | Terminal value              | Payoff                    |
        |-----------------------------|---------------------------|
        | S(T) $\le$ H                | S                         |
        | H< S(T) $\le$ S_0           | S_0*(TW*(-(S-S_0)/S_0))   |
        | S_0 < S(T) $\le$ S_0*(1+MR) | S                         |
        | S(T)>S_0*(1+MR).            | S_0*(1+MR)                |
    
        """
        
        filename, update = self.initialize_test(update)  
        T = 3.5
        H = (1-0.2250)*self.S_0
        MR = 0.28
        TW = 1
        
        
        
        self.register(T, barriers=[H])
        # A structured producted is often referred to as a "note"
        self.note_spotput = Spot(self.S_0) + (1+TW)*Put(self.S_0, self.K) - (1+TW)*Put(self.S_0, H)  
        self.note_digitalput= (1+TW)*(self.K-H)*DigitalPut(np.nan, H) + Call(self.S_0, (1+MR)*self.K) 

        self.note = self.note_spotput - self.note_digitalput 
        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
      
    
    
    def test_nbc1861(self,update=None):
        """NBC1861 NBC iShares® S&P/TSX 60 Index ETF Deposit Notes S1
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1242

        * Payoff table: 

        | Terminal value  | Payoff |
        |-----------------|--------|
        | S(T) $\le$ K    | S_0    |
        | S(T)>K          | PF*S   |
        """
        
        filename, update = self.initialize_test(update)  
        T = 5
        H = (1-0)*self.S_0       
        PF = 0.7
        
        self.register(T, barriers=[H])
        self.note  = Spot(self.S_0) - (1-PF)*Call(self.S_0, self.K) + Put(self.S_0, self.K)    
        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
        
        
    def test_nbc1762(self,update=None):
        """NBC1762  NBC Fixed ROC Note Securities (Maturity-Monitored Barrier) linked to the units of the iShares® S&P/TSX 60 Index ETF
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1297

        # NBC1762 NBC1778
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1297 \\
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1723

        * Payoff table: 

        | Terminal value            | Payoff                    |
        |---------------------------|---------------------------|
        | S(T) $\le$ H              | S + coupon                |
        | H< S(T) $\le$ S_0         | S_0  + coupon             |
        | S_0< S(T) $\le$ S_0*(1+RT)| S_0*(1+RT) +coupon, RT=0  |
        | S(T)>S_0*(1+RT)  .        | S_0+PF*(S-S_0) + coupon   |
        """
        
        filename, update = self.initialize_test(update)
        
        T = 5.5
        H = (1-0.25)*self.S_0
        PF = 0.05
        RT = 0.22
        K_1 = self.S_0
        K_2 = (1+ RT)*self.S_0
        T_1 = 0.5
        ROC = 2
        
        
        
        self.register(T, barriers=[H])
        # A structured producted is often referred to as a "note"
        self.note_put =   Put( self.S_0, K_1) - Put( self.S_0, H) - (K_1-H)*DigitalPut(np.nan,H) - Call(self.S_0, K_1) 
        self.note_bond_call =Spot( self.S_0)+  PF*Call( self.S_0,  K_2) +  ROC*Bond( T)


        self.note  =  self.note_put+self.note_bond_call 
        results=self.note.payoff(self.S_T) 
        
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc1287(self,update=None):
        """NBC1287, NBC Marathon Note Securities (Maturity-Monitored Barrier) linked to the units of the iShares® S&P/TSX 60 Index ETF
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1308


        * Payoff table: 

        | Terminal value    | Payoff |
        |-------------------|--------|
        | S(T) $\le$ H      | S      |
        | H< S(T) $\le$ S_0 | S_0    |
        | S(T)>S_0.         | AF*S   |
        """
        
        filename, update = self.initialize_test(update)  
        T = 5
        H = (1-0.25)*self.S_0
        
        AF = 1.525
        self.register(T, barriers=[H], acceleration=AF)
        
        # A structured producted is often referred to as a "note"
        self.note_put = Put(self.S_0, self.K) - Put(self.S_0, H) - (self.K-H)*DigitalPut(np.nan, H)  
        self.note_spot_call = Spot(self.S_0) +(AF-1)*Call(self.S_0, self.K)


        self.note  =  self.note_put+self.note_spot_call 
        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
        
        
    def test_nbc1288(self,update=None):
        """NBC1288, NBC Marathon Note Securities (Buffered) linked to the units of the iShares® S&P/TSX 60 Index ETF
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1331


        * Payoff table: 

        | Terminal value    | Payoff        |
        |-------------------|---------------|
        | S(T) $\le$ H      | S+ S_0*buffer |
        | H< S(T) $\le$ S_0 | S_0           |
        | S(T)>S_0.         | AF*S          |
    
        """
        
        filename, update = self.initialize_test(update)  
        T = 5.5
        H = (1-0.25)*self.S_0
        BF = 0.25
        
        AF = 1.10
        
        
        self.register(T, barriers=[H], acceleration=AF)
        # A structured producted is often referred to as a "note"
        self.note =Spot(self.S_0) + Put(self.S_0, self.K) - Put(self.S_0, H) + (AF-1)*Call(self.S_0, self.K)

        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
       

    
    def test_nbc1681(self,update=None):
        """NBC1681, NBC Bonus Note Securities (Maturity-Monitored Barrier) linked to the Canadian market
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1413


        * Payoff table: 

        | Terminal value                       | Payoff                 |
        |--------------------------------------|------------------------|
        | S(T) $\le$ H                         | S                      |
        | H< S(T) $\le$ S_0*(1+Booster return) | S_0*(1+Booster return) |
        | S(T)>S_0*(1+Booster return) .        | PF*S                   |
    
        """
        
        filename, update = self.initialize_test(update)  
        T = 5
        H = (1-0.1)*self.S_0
        BR = 0.4
        
        PF = 1
        
        self.register(T, barriers=[H])
        # A structured producted is often referred to as a "note"
        self.note_digital_put = Put(self.S_0, (1+BR)*self.K) - Put(self.S_0, H) - ((1+BR)*self.K-H)*DigitalPut(np.nan, H) 
        self.note_call = Spot(self.S_0)  +(1-PF)*Call(self.S_0, (1+BR)*self.K)
        
        self.note=self.note_call+self.note_digital_put
        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
        

    
    def test_nbc1869(self,update=None):
        """NBC1869 NBC iShares® S&P/TSX 60 Index ETF Deposit Notes
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1318

        * Payoff table: 

        | Terminal value | Payoff |
        |----------------|--------|
        | S(T) $\le$ K   | S_0    |
        | S(T)>K         | PF*S   |
        """
        
        filename, update = self.initialize_test(update)  
        
        # K should be arithmetic average (expressed as a percentage and rounded to two decimal places) 
        # of the price return of the Reference Shares over the period starting on the Issue Date of 
        # # the NBC iShares® S&P/TSX 60 Index ETF Deposit Notes and ending on the Valuation Date.
        T = 5.5
        H = (1-0)*self.S_0
        
        PF = 0.7
        
        
        self.register(T, barriers=[H])
        
        # A structured producted is often referred to as a "note"
        
        self.note = Spot(self.S_0) - (1-PF)*Call(self.S_0, self.K) + Put(self.S_0, self.K) 
        results=self.note.payoff(self.S_T) 
       
        
        self.finalize_test(results, filename, update)
        

        
    
    def test_nbc2305(self,update=None):
        """NBC2305, NBC Marathon Note Securities (Buffered) linked to the Canadian market
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1671


        * Payoff table: 

        | Terminal value    | Payoff       |
        |-------------------|--------------|
        | S(T) $\le$ H      | S+ S_0*buffer|
        | H< S(T) $\le$ S_0 | S_0          |
        | S(T)>S_0.         | AF*S         |
        """
        
        filename, update = self.initialize_test(update)  
        # K should be arithmetic average (expressed as a percentage and rounded to two decimal places) 
        # of the price return of the Reference Shares over the period starting on the Issue Date of 
        # # the NBC iShares® S&P/TSX 60 Index ETF Deposit Notes and ending on the Valuation Date.
        T = 6
        H = (1-0.30)*self.S_0
        BF = 0.30
        
        AF = 1.20
        
        self.register(T, barriers=[H], acceleration=AF)
        # A structured producted is often referred to as a "note"
        
        self.note = Spot(self.S_0) + Put(self.S_0, self.K) - Put(self.S_0, H) + (AF-1)*Call(self.S_0, self.K)
        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc2307(self,update=None):
        """NBC2307 NBC Marathon Note Securities (Buffered) linked to the Canadian market
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1683

        * Payoff table: 

        | Terminal value    | Payoff       |
        |-------------------|--------------|
        | S(T) $\le$ H      | S+ S_0*buffer|
        | H< S(T) $\le$ S_0 | S_0          |
        | S(T)>S_0.         | AF*S         |
        """
        
        filename, update = self.initialize_test(update)  
        # K should be arithmetic average (expressed as a percentage and rounded to two decimal places) 
        # of the price return of the Reference Shares over the period starting on the Issue Date of 
        # # the NBC iShares® S&P/TSX 60 Index ETF Deposit Notes and ending on the Valuation Date.
        T = 5
        H = (1-0.25)*self.S_0
        AF = 1.125
        BF = 0.25
               
        self.register(T, barriers=[H], acceleration=AF)
        # A structured producted is often referred to as a "note"
        
        self.note = Spot(self.S_0) + Put(self.S_0, self.K) - Put(self.S_0, H) + (AF-1)*Call(self.S_0, self.K)
        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc2317(self,update=None):
        """NBC2317 NBC Marathon Note Securities (Maturity-Monitored Barrier)
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1859

        * Payoff table: 

        | Terminal value    | Payoff |
        |-------------------|--------|
        | S(T) $\le$ H      | S      |
        | H< S(T) $\le$ S_0 | S_0    |
        | S(T)>S_0.         | AF*S   |
        """
        
        filename, update = self.initialize_test(update)  
        # K should be arithmetic average (expressed as a percentage and rounded to two decimal places) 
        # of the price return of the Reference Shares over the period starting on the Issue Date of 
        # # the NBC iShares® S&P/TSX 60 Index ETF Deposit Notes and ending on the Valuation Date.
        T = 6
        H = 0.50*self.S_0
        
        AF = 1.15
      
        self.register(T, barriers=[H], acceleration=AF)
        # A structured producted is often referred to as a "note"
        self.note_spot_call =Spot(self.S_0) + (AF-1)*Call(self.S_0, self.K)
        self.note_digital_put =  Put(self.S_0, self.K) - Put(self.S_0, H) - (self.K-H)*DigitalPut(np.nan, H)
        self.note =self.note_spot_call + self.note_digital_put 

        
        results=self.note.payoff(self.S_T) 
        
        
        self.finalize_test(results, filename, update)
        
        
    
    def test_nbc2318(self,update=None):
        """NBC2318, NBC Marathon Note Securities (Maturity-Monitored Barrier) 
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1873


        * Payoff table: 

        | Terminal value    | Payoff |
        |-------------------|--------|
        | S(T) $\le$ H      | S      |
        | H< S(T) $\le$ S_0 | S_0    |
        | S(T)>S_0.         | AF*S   |
        """
        
        filename, update = self.initialize_test(update)  
        T = 6
        H = (1-0.4)*self.S_0
        
        AF = 1.67

        self.register(T, barriers=[H], acceleration=AF)
        # A structured producted is often referred to as a "note"
        self.note_spot_call =Spot(self.S_0) + (AF-1)*Call(self.S_0, self.K)
        self.note_digital_put = Put(self.S_0, self.K) - Put(self.S_0, H) - (self.K-H)*DigitalPut(np.nan, H) 

        self.note =self.note_spot_call + self.note_digital_put 
      
        results=self.note.payoff(self.S_T) 
       
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc2321(self,update=None):
        """NBC2321, NBC Marathon™ Note Securities (No Barrier)
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1960
        # NBC2324
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=1960


        * Payoff table: 

        | Terminal value | Payoff|
        |----------------|-------|
        | S(T) $\le$ S_0 | S     |
        | S(T)>S_0.      | AF*S  |

        """
        
        filename, update = self.initialize_test(update)  
        T = 5
        H = (1-0.0)*self.S_0
        
        AF = 2.25
        
        self.register(T, barriers=[H], acceleration=AF)
        # A structured producted is often referred to as a "note"
        self.note_spot_call =Spot(self.S_0) + (AF-1)*Call(self.S_0, self.K)
        self.note_digital_put =  Put(self.S_0, self.K) - Put(self.S_0, H) - (self.K-H)*DigitalPut(np.nan, H) 

        self.note =self.note_spot_call + self.note_digital_put 
      
        results=self.note.payoff(self.S_T)
        
        
        self.finalize_test(results, filename, update)
        

    def test_nbc2542(self,update=None):
        """NBC2542, NBC Marathon™ (Accelerator) Note Securities (Maturity-Monitored Barrier) linked to the Canadian market
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=2681


        * Payoff table: 

        | Terminal value   | Payoff |
        |------------------|--------|
        | S(T) $\le$ H     | S      |
        | H< S(T) $\le$ S_0| S_0    |
        | S(T)>S_0.        | AF*S   |
        """
        
        filename, update = self.initialize_test(update)  
        # K should be arithmetic average (expressed as a percentage and rounded to two decimal places) 
        # of the price return of the Reference Shares over the period starting on the Issue Date of 
        T = 5.5
        H = (1-0.3)*self.S_0
        
        AF = 1.45
        
        self.register(T, barriers=[H], acceleration=AF)
        # A structured producted is often referred to as a "note"
        self.note_spot_call =Spot(self.S_0) + (AF-1)*Call(self.S_0, self.K)
        self.note_digital_put = Put(self.S_0, self.K) - Put(self.S_0, H) - (self.K-H)*DigitalPut(np.nan, H) 


        self.note =self.note_spot_call + self.note_digital_put 
      
        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
        
        
    
    def test_nbc26740(self,update=None):
        """NBC26740, NBC S&P/TSX Composite Low Volatility Index Deposit Notes 
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=4764


        * Payoff table: 

        | Terminal value | Payoff |
        |----------------|--------|
        | S(T) $\le$ S_0 | S_0    |
        | S(T)>S_0.      | AF*S（T）|
        """
        
        filename, update = self.initialize_test(update)  
        T = 5
        AF = 1.75       
        
        self.register(T,  acceleration=AF)
        # A structured producted is often referred to as a "note"
        self.note = AF*Call(self.S_0, self.K) + self.K*Bond(T)
        
        results=self.note.payoff(self.S_T) 
        
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc26741(self,update=None):
        """NBC26741, NBC Canadian Market Low Volatility GIC
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=4826

        * Payoff table: 

        | Terminal value  | Payoff |
        |-----------------|--------|
        | S(T) $\le$ S_0  | S_0    |
        | S(T)>S_0.       | AF*S（T）|

        """
        
        filename, update = self.initialize_test(update)  
        T = 5
        AF = 1.75
      
        self.register(T, acceleration=AF)
        # A structured producted is often referred to as a "note"
        self.note = AF*Call(self.S_0, self.K) + self.K*Bond(T)        
        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc26742(self,update=None):
        """NBC26742， NBC American companies GIC
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=4779


        * Payoff table: 

        | Terminal value      | Payoff    |
        |---------------------|-----------|
        | S(T) $\le$ S_0      | S_0       |
        | S_0<S(T)<S_0*(1+MR).| S（T）      |
        | S_0<S(T)>S_0*(1+MR).| S_0*(1+MR)|
        """
        
        filename, update = self.initialize_test(update)  
        
        T = 5
        TW = 1
        MR = 0.25 # Maximum Variable Return
        
        
        self.register(T)
        # A structured producted is often referred to as a "note"
        self.note = Spot(self.S_0) + Put(self.S_0, self.K) - Call(self.S_0, (1+MR)*self.K)
        results=self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc26743(self,update=None):
        """NBC26743
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=4827


        * Payoff table: 

        | Terminal value  | Payoff  |
        |-----------------|---------|
        | S(T) $\le$ S_0  | S_0     |
        | S(T)>S_0.       | PF*S（T） |
        """
        
        filename, update = self.initialize_test(update)  
        T = 2.5
        PF = 1
      
        self.register(T)
        # A structured producted is often referred to as a "note"
        self.note = Spot(self.S_0) + PF*Put(self.S_0, self.K)
        results=self.note.payoff(self.S_T) 
        
        
        self.finalize_test(results, filename, update)
        
    def test_nbc26744(self,update=None):
        """NBC26744  NBC S&P/TSX Composite Low Volatility Index with Low Point Deposit Notes (USD)
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=4784
        # NBC26745 NBC26790 NBC26799 NBC26804
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=4785
        # NBC26745 Return from low point over first 9 months (observed daily)
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=4823
        # NBC26813 NBC26819 NBC26825 NBC26830 NBC26832 NBC26834 NBC26837 NBC26838

        * Payoff table: 

        | Terminal value                         | Payoff                                    |
        |----------------------------------------|-------------------------------------------|
        | S(T) $\le$ Initial level (lowest point)| S_0                                       |
        | S(T)>Initial level (lowest point).     | S_0*max(0, (S-Lowest price))/Lowest price |

        """
        
        
        filename, update = self.initialize_test(update) 
        T = 6
        K_1 = 0.6*self.S_0
        K_2 = self.S_0
        
        PF = 1  
       
        self.register(T)
        # A structured producted is often referred to as a "note"
        self.note_1 = (K_2/K_1)*Spot(self.S_0) + (K_2/K_1)*Put(self.S_0, K_1) + (PF-1) *(K_2/K_1)*Call(self.S_0, K_1) 
        self.note_2 = Spot(self.S_0) + Put(self.S_0, K_2) + (PF-1)*Call(self.S_0, K_2)
 
        results=np.concatenate(( self.note_1.payoff(self.S_T) , self.note_2.payoff(self.S_T) ))    
       
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc26746(self,update=None):
        """NBC26746, NBC Canadian Market Low Volatility Flex GIC
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=4799 
        # 26747
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=4810

        * Payoff table: 

        | Terminal value | Payoff |
        |----------------|--------|
        | S(T) $\le$ S_0 | S_0    |
        | S(T)>S_0.      | S（T）   |

        """
        
        
        filename, update = self.initialize_test(update)  
        
        T = 2.5
        PF = 1
        
        self.register(T)
        # A structured producted is often referred to as a "note"
        self.note = Spot(self.S_0) + PF*Put(self.S_0, self.K)
 
        results = self.note.payoff(self.S_T) 
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc26779(self,update=None):
        """NBC26779 NBC Canadian Market Low Volatility with Low Point Flex
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=5010

        * Payoff table: 

        | Terminal value                         | Payoff                                                           |
        |----------------------------------------|------------------------------------------------------------------|
        | S(T) $\le$ Initial level (lowest point)| S_0                                                              |
        | S(T)>Initial level (lowest point).     | S_0\*participation factor\*max(0, (S-Lowest price))/Lowest price |
        """
        
        
        filename, update = self.initialize_test(update) 
        T = 6.5
        K_1 = 0.6*self.S_0
        K_2 = self.S_0
        
        PF = 1.1  
        
        self.register(T)
        # A structured producted is often referred to as a "note"
        self.note_1 = ( K_2/K_1)*Spot(self.S_0) + (K_2/K_1)*Put(self.S_0, K_1) + (PF-1) *(K_2/K_1)*Call(self.S_0, K_1) 
        self.note_2 = Spot(self.S_0) + Put(self.S_0, K_2) + (PF-1)*Call(self.S_0, K_2)
 
        results=np.concatenate(( self.note_1.payoff(self.S_T) , self.note_2.payoff(self.S_T) ))
       
        
        self.finalize_test(results, filename, update)
        
    
    def test_nbc26808(self,update=None):
        """NBC26808 NBC Canadian Market Low Volatility with Low Point Flex GIC
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=5235

        * Payoff table: 

        | Terminal value                         | Payoff                                                           |
        |----------------------------------------|------------------------------------------------------------------|
        | S(T) $\le$ Initial level (lowest point)| S_0                                                              |
        | S(T)>Initial level (lowest point).     | S_0\*participation factor\*max(0, (S-Lowest price))/Lowest price |

        """
        
        
        filename, update = self.initialize_test(update)  
        T = 8
        K_1 = 0.6*self.S_0
        K_2 = self.S_0
        PF = 1.65
      
      
        self.register(T)
        # A structured producted is often referred to as a "note"
        self.note_1 = (self.S_0/K_1)*Spot(self.S_0) + (self.S_0/K_1)*Put(self.S_0, K_1) + (PF-1)*(self.S_0/K_1)*Call(self.S_0, K_1) 
        self.note_2 = Spot(self.S_0) + Put(self.S_0, K_2) + (PF-1)*Call(self.S_0, K_2) 
 
        results=np.concatenate(( self.note_1.payoff(self.S_T) , self.note_2.payoff(self.S_T) ))
       
        
        self.finalize_test(results, filename, update)
        
        
    def test_nbc25227(self,update=None):
        """NBC25227, NBC Marathon™ (Accelerator) Note Securities (Maturity-Monitored Barrier) linked to the Canadian market,
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=5475

        * Payoff table: 

        | Terminal value    | Payoff |
        |-------------------|--------|
        | S(T) $\le$ H      | S      |
        | H< S(T) $\le$ S_0 | S_0    |
        | S(T)>S_0.         | AF*S   |
        """
        
        filename, update = self.initialize_test(update) 
        T = 5
        H = (1-0.3)*self.S_0
       
        AF = 2.25
        
        self.register(T, barriers=[H], acceleration=AF)
        # A structured producted is often referred to as a "note"
        self.note = Spot(self.S_0) + Put(self.S_0, self.K) - Put(self.S_0, H) - (self.K-H)*DigitalPut(np.nan, H) + (AF-1)*Call(self.S_0, self.K)
 
        results= self.note.payoff(self.S_T) 
       
        
        self.finalize_test(results, filename, update)
       
    def test_nbc25230(self,update= None):
        """NBC25230, NBC Marathon™ (Accelerator) Note Securities (No Barrier) linked to the Canadian market
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=5476

        * Payoff table: 

        | Terminal value   | Payoff |
        |------------------|--------|
        | S(T) $\le$ S_0   | S      |
        | S(T)>S_0.        | AF*S   |
        """
        
        
        filename, update = self.initialize_test(update) 
        T = 5
        AF = 2.75
        
        self.register(T,  acceleration=AF)
        # A structured producted is often referred to as a "note"
        self.note = Spot(self.S_0) + (AF-1)*Call(self.S_0, self.K)
        
        results= self.note.payoff(self.S_T) 
       
        
        self.finalize_test(results, filename, update)
        
    def test_nbc20283(self,update=None):
        """NBC20283, NBC Fixed Coupon Note Securities (Maturity-Monitored Barrier) linked to the Canadian market
        https://nbcstructuredsolutions.ca/detailProduit.aspx?lequel=5590


        * Payoff table: 

        | Terminal value   | Payoff               |
        |------------------|----------------------|
        | S(T) $\le$ H     | S + coupon           |
        | H< S(T) $\le$ S_0| S_0  + coupon        |
        | S(T)>S_0.        | S+PF*(S-S_0) + coupon|
        """
        
        
        filename, update = self.initialize_test(update)  
        T = 6
        H = (1-0.30)*self.S_0
        
        PF = 0
        RT = 0
        
        ROC = 2.5
        
        self.register(T, barriers=[H])
        # A structured producted is often referred to as a "note"
        self.note_digital_put =  Put(self.S_0, self.K) - Put(self.S_0, H) - (self.K-H)*DigitalPut(np.nan, H) 
        self.note_spot_call = Spot(self.S_0) + (PF-1)*Call(self.S_0, self.K) + ROC*Bond(T)
        self.note = self.note_spot_call + self.note_digital_put
        results= self.note.payoff(self.S_T) 
       
        
        self.finalize_test(results, filename, update)
        

           

class AllTests(unittest.TestCase):
    def suite():
        """Create a test suite containing all test cases from both modules."""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        '''
        # Manually add test cases in the desired order
        test_cases = [TestNBC1155,TestV75863,TestNBC1280,TestNBC1282,TestNBC2709,TestNBC1069, 
                      TestNBC1861,TestNBC1762NBC1778,TestNBC1287,TestNBC1288,TestNBC1681,TestNBC1869,
                      TestNBC2305,TestNBC2307,TestNBC2317,TestNBC2318,TestNBC2324,TestNBC2542,TestNBC26740,
                      TestNBC26741,TestNBC26742,TestNBC26743,TestNBC26744,TestNBC26746,TestNBC26779,TestNBC26808,
                      TestNBC25227,TestNBC25230,TestNBC20283]
        '''
        # Manually add test cases in the desired order
        test_cases = [
            TestStructuredProducts]
        for test_case in test_cases:
            suite.addTests(loader.loadTestsFromTestCase(test_case))
        return suite


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args)==1 and args[0]=='--update':
        TestStructuredProducts.update = True
        
    runner = unittest.TextTestRunner()
    runner.run(AllTests.suite())


#if __name__ == '__main__':
#unittest.main()
