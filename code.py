"""
Version: 29 June 2022

Authors: Michaela Cully-Hugill and Ethan S. Lee

This is supplementary code for the computations in the paper "Explicit Interval Estimates for Prime Numbers" by M. Cully-Hugill and E. S. Lee (publishedin Math. Comp. at https://doi.org/10.1090/mcom/3719).
An arXiv version of the paper (including corrigendum) is available at https://arxiv.org/abs/2103.05986.

Parameter variables (in brackets is the corresponding laTex label in the paper): m, n, d (\delta), a, x0 (x_0), y0 (is exponent of x0 = e^{y0}), T0 (T_0), T1 (T_1), s (\sigma_0)
Note that we reference equations and theorem numbers from the published paper.
"""

import mpmath as mpm
from mpmath import mpf, nstr, fprod, factorial, fmul, fdiv, exp, log, sqrt, pi, power, inf

from scipy.optimize import differential_evolution, NonlinearConstraint
from scipy.integrate import quad

from math import comb, floor
from fractions import Fraction

mpm.mp.dps = 350        #Number of decimal places used when storing an mpf value
                        #Can be reduced to 100 for small x0, but needs to be increased up to around 350 for larger x0, say x0=exp(600), particularly when 'complex-value' or NaN errors are returned

R0 = mpf(5.573412)       # Zero-free region constant $R_0$
H = 3000175332800   # Riemann height
T0 = 104537615      # $T_0$, the largest imaginary part up to which we have counted $N_0$ zeros
N0 = 2.6*10**8      # $N_0$, the number of zeros with ordinates in (0, $T_0$]
S0 = 21.98308       # $S_0$, an upper bound on the sum of reciprocal ordinates (i.e. $1/\frac{1}{\gamma}$) in (0, $T_0$]

w = mpf(1.0344*10**(-3))     # \omega
a1 = mpf(0.11)               # a_1
a2 = mpf(0.29)               # a_2
a3 = mpf(2.29)               # a_3
A0 = mpf(2.067)              # A_0
A1 = mpf(0.059)              # A_1
A2 = mpf(1/150)              # A_2

A  = 58773/10000            # A from Kadiri, Lumley, and Ng's zero-density estimate
B  = 3869/1000              # B from Kadiri, Lumley, and Ng's zero-density estimate

def u(d,m):                 
    # u = \delta/m
    return fdiv(d,m)

def Delta(x):               
    # \Delta; returns -\Delta because the optimiser seeks to minimise the function, so we will maximise Delta by minimising -\Delta
    m=floor(x[0])
    d=mpf(x[1])
    a=x[2]
    return -(1 - 1/(exp(u(d,m))*(1 + d*(1-a))) - (d*a)/(exp(u(d,m))*(1 + d*(1-a))))**(-1)     # Defined as the negative because the optimiser seeks to minimise the function, so we will maximise Delta by minimising -Delta

def X0(m, d, a, y0):        
    # X_0; returned value is equivalent to x0/(exp(d/m)*(1 + d(1-a)))
    return exp(y0 - u(d,m) - log(1 + d*(1-a))) 
    
def An(n):                  
    # Constant A in the smoothing function, f_2
    return Fraction((n + 1)**(n + 1)/n**n)

def prod1(m,n):
    return fprod(range(m*n + 1, m*n + m + 2))

def fm1(m,n):               
    # ||f||_1
    return power(An(n),m) * factorial(m)/prod1(m,n)

def vf(m,n,a):              
    # \nu(f,a), the sum of two integrals of f_2 over t\in(0,a) and t\in (1-a, 1)
    sumV = 0
    for k in range(m):
        sumV = sumV + fdiv(fprod(range(m-k+1,m+1)), fprod(range(m*n+1, m*n+k+2))) * ((1-a)**(m-k)*a**(m*n+k+1) - a**(m-k)*(1-a)**(m*n+k+1))
    return power(An(n),m)*sumV + fm1(m,n)*(1 - (1-a)**(m*n+m+1) + a**(m*n+m+1))

def L1(m, n, d):            
    # \lambda_1 in Lemma 7, the upper bound on F(1,m,\delta)
    Bn = fdiv(n, n + 1)
    return 2*(1 + d)**2*power(power(Bn, n) - power(Bn, n + 1), m)*prod1(m,n)/factorial(m)

def L(m,n,d):               
    # \lambda in Lemma 7, the upper bound on F(m,m,\delta)
    def prod01(m,n,k):
        return fprod(range(m*n + k - 2*m + 1, m*n + k + 1))
    def prod02(m,n,k):
        return fprod(range(2*m*n + k - 2*m + 1, 2*m*n + k - m + 2))
    sum2 = 0
    for k in range(m+1):
        sum2 = sum2 + (-1)**(m+k) * comb(m, k) * prod01(m,n,k)/prod02(m,n,k)
    return power(fdiv(power(1 + d, 2*m+3) - 1, d*(2*m+3)), 0.5) * prod1(m,n)/sqrt(factorial(m)) * power(sum2, 0.5)

def ZF(T):                  
    # The function in the zero-free region, see (2), R0 (R_0) is the constant
    return 1/(R0*log(T))
def R(T):                   
    # The upper bound on |N(T) - P(T)|; note that we use the bound T\leq H in second term
    return a1*log(T) + a2*log(log(H)) + a3 
def P(T):
    # Main term in the estimate for the zero-counting function N(T)
    return T/(2*pi)*log(T/(2*pi)) - T/(2*pi) + 7/8 + R(T)

def ZD(s, T):
    # The zero-density estimate in (5)
    return A*log(T)**(5 - 2*s)*T**(8/3*(1 - s)) + B*(log(T))**2
def ZDderv(s, T):
    # The derivative of ZD(s, T)
    return (5 - 2*s)*A*log(T)**(4 - 2*s)*T**(8/3*(1 - s)-1) + A*log(T)**(5 - 2*s)*(8/3*(1 - s))*T**(8/3*(1 - s)-1) + 2*B*log(T)/T

# S1, ..., S5 are the functions S_1, ..., S_5, defined in Theorem 2
def S1(T):
    E1 = 2*(A0 + A1*log(T0))/T0**2 + (A1 + A2)/T0**2
    return log(T/T0)*log(sqrt(T*T0)/(2*pi))/(2*pi) + R(T0)/T0 + (R(T) + 0.5)/T + E1
def S2(m, T):
    E2 = fdiv(2*(m+1)*(A0 + A1*log(T)), power(T,m+2)) + fdiv((A1 + A2), power(T,m+2))
    return (1 + m*log(T/(2*pi)))/(2*pi*m**2*power(T,m)) - (1 + m*log(H/(2*pi)))/(2*pi*m**2*power(H,m)) + fdiv(R(T), power(T,m+1)) + fdiv(R(H)+0.5, power(H, m+1)) + E2
def S3(m):
    E3 = fdiv(2*(m+1)*(A0 + A1*log(H)), power(H,m+2)) + fdiv((A1 + A2), power(H,m+2))
    return (1 + m*log(H/(2*pi)))/(2*pi*m**2*power(H,m)) + fdiv(R(H), power(H,m+1)) + E3
def S4(m,s):
    dZD = lambda t: ((5 - 2*s)*A*log(t)**(4 - 2*s)*t**(8/3*(1 - s)-1) + A*log(t)**(5 - 2*s)*(8/3*(1 - s))*t**(8/3*(1 - s)-1) + 2*B*log(t)/t) * t**(-(m + 1))
    #return 2*ZD(s, H)/H**(m + 1)           # This is a simplified expression for S4 if using a higher Riemann height
    return fdiv(ZD(s, H), power(H, m + 1)) + sum(quad(dZD, H, inf))
def S5(X,m,s):
    dZD = lambda t: ((5 - 2*s)*A*log(t)**(4 - 2*s)*t**(8/3*(1 - s)-1) + A*log(t)**(5 - 2*s)*(8/3*(1 - s))*t**(8/3*(1 - s)-1) + 2*B*log(t)/t) * t**(-(m + 1))
    #return 2*X**(-1/(R0*log(H)))*ZD(s, H)/H**(m + 1)       # This is a simplified expression for S5 if using a higher Riemann height
    return power(X, -ZF(H))*fdiv(ZD(s, H), power(H, m + 1)) + sum(quad(dZD, H, inf))

def F0m(m,n,d):
    # F(0,m,\delta)
    return 1 + (m*n+1)*d/(m*n+m+2)

# Sum01, Sum02, Sum11, Sum12, are functions used in defining the following B_0, ..., B_{42} functions
def Sum01(m, n, d):
    return 4*L1(m,n,d)/((exp(u(d,m)/2) + 1)*d)*S0
def Sum02(m,n,d):
    return 4*F0m(m,n,d)/(exp(u(d,m)/2) + 1)*N0
def Sum11(m,n,d,T):
    return 4*L1(m,n,d)/((exp(u(d,m)/2) + 1)*d)*S1(T)
def Sum12(m,n,d,T):
    return 4*F0m(m,n,d)/(exp(u(d,m)/2) + 1)*(P(T) + R(T) - N0)

# B0, ..., B42 are the functions B_0, ..., B_{42}, defined in Theorem 2
def B0(m,n,d):
    return min(Sum01(m,n,d),Sum02(m,n,d))
def B1(m,n,d,T):
    return min(Sum11(m,n,d,T), Sum12(m,n,d,T))
def B2(m,n,d,T):
    return fdiv(2*L(m,n,d)*S2(m, T),fmul((exp(d/(2*m)) - 1), d**m))
def B3(m,n,d,s):
    return fdiv(2*L(m,n,d)*S3(m)*fdiv((exp(u(d,m)*s) + 1), (exp(u(d,m)) - 1)), d**m)
def B41(X,m,n,d,s):
    return fdiv(2*L(m,n,d)*S5(X, m, s)*fdiv((exp(u(d,m)) + 1), (exp(u(d,m)) - 1)), d**m)
def B42(m,n,d,s):
    return fdiv(2*L(m,n,d)*S4(m, s)*fdiv((exp(u(d,m)) + 1), (exp(u(d,m)) - 1)), d**m)

# Sum0, ..., Sum4 are individual terms in eq (9) of Theorem 2
def Sum0(m, n, d, a, y0):
    return B0(m,n,d)*X0(m, d, a, y0)**(-1/2)
def Sum1(m, n, d, a, T, y0):
    return B1(m,n,d,T)*X0(m, d, a, y0)**(-1/2)
def Sum2(m, n, d, a, T, y0):
    return B2(m,n,d,T)*X0(m, d, a, y0)**(-1/2)
def Sum3(m, n, d, a, s, y0):
    return B3(m,n,d,s)*X0(m, d, a, y0)**(s - 1) + B3(m,n,d, 1 - s)*X0(m, d, a, y0)**(-s)
def Sum4(m, n, d, a, s, y0):
    return B41(X0(m, d, a, y0), m, n, d, s)*X0(m, d, a, y0)**(-ZF(H)) + B42(m,n, d, s)*X0(m, d, a, y0)**(-1 + ZF(H)) 

def TheSum(m, n, d, a, T, s, y0):
    # The terms in eq (9) which result from using the Riemann-von Mangoldt explicit formula
    result = Sum0(m, n, d, a, y0) + Sum1(m, n, d, a, T, y0) + Sum2(m, n, d, a, T, y0) + Sum3(m, n, d, a, s, y0) + Sum4(m, n, d, a, s, y0)
    return result

def BrunTerms(m, n, d, a, y0):
    # The terms in eq (9) which result from using the Brun-Titchmarsh theorem
    result = (exp(u(d,m)) - 1)**(-1)*(u(d,m)/2*X0(m, d, a, y0)**(-2) + w*X0(m, d, a, y0)**(-1/2)) + 2*vf(m,n,a)/fm1(m,n)*(1 + d)*(u(d,m) + log((1+d)*X0(m, d, a, y0)))/log((1+d)*(exp(u(d,m)) - 1)*X0(m, d, a, y0))
    return result

def constraint1(p): 
    """
    Returns the left-hand side of Theorem 2. Input is a list of parameter values p = [m, d, a, T, n, s].
    If this returns a positive value for some p, those parameter values define a valid pair of \Delta and x_0 in Theorem 1.
    Note that x_0 is defined by x0 below. To check the table of results (values given at the end of this code) with the given loop, change the function to constraint1(p,x0) and comment out fixed x0
    """
    m=floor(p[0])
    d=mpf(p[1])
    a=p[2]
    T=mpf(p[3])
    n=floor(p[4])
    s=s0
    y0=log(x0) #y0 is the exponent of x0 = exp(y0)
    return F0m(m,n,d) - TheSum(m, n, d, a, T, s, y0) - BrunTerms(m, n, d, a, y0)     # Returned value is f(x) in the condition f(x)\geq 0


x0 = exp(60)       # Is x_0 for a result holding for all x\geq x_0
s0 = Fraction(7804/10000)   # \sigma_0

##-------------To print the constraint1 and \Delta for specific parameter values-----------

m0=30 
d0=4.92432301e-11 
a0=0.3846585
T1= 1e+11
n0=1
# j = [28, 7.435876822271507e-11 , 0.5771 , 956059635018 , 2] #[m0, d0, a0, T1, n0]
# print(round(constraint1(j),7), nstr(-Delta(j),5))

##------------------------------------------------------------------------------------------


##----------------------------Differential evolution method--------------------------------
## This section contains the differential evolution optimiser. The list 'bounds' may need to be adjusted when considering larger or smaller x0.
## Note the dps for mpmath may need to be increased for larger x0 - can use 100 for smallest x0, and may need up to 400 for larger x0.

nlc = NonlinearConstraint(constraint1, 0, inf)  # Specifies that we want constraint1 to be greater than 0
bounds = [(50,140), (10**(-13),10**(-10)), (0.3,0.5), (T0,H), (1,1)] # Range of m, \delta, a, T_1, and n we want the optimiser to search within

Nfeval = 1
def callbackF(Xi, convergence):
    # The function callback is called by the differential evolution method. 
    # It prints the value of the constraint function (constraint1) and \Delta for each iteration, and the corresponding set of parameter values, rounded to specific decimal places.
    global Nfeval
    print('Constraint =', round(constraint1(Xi),6), ', Delta =', nstr(-Delta(Xi),5))
    print('m =', floor(Xi[0]), ', delta =', Xi[1], ', a =', round(Xi[2],6), ', T =', round(Xi[3]), ', n =', floor(Xi[4]), "\n")
    Nfeval += 1

result = differential_evolution(Delta, bounds, constraints=(nlc), seed=1, disp=True, callback=callbackF, polish=False)

print(result.x, result.fun)     # Prints the final evalution of the differential evolution method



##--------------Checking parameter values in table of results-------------
## Uncomment this section to compute the constraint and Delta for each set of optimised parameter values
## Will need to change definition of constraint1 and use sufficienty high dps in mpmath (e.g. 350)

# Xo=[4*10**18, exp(43), exp(46), exp(50), exp(55), exp(60), exp(75), exp(90), exp(105), exp(120), exp(135), exp(150), exp(300), exp(600)]

# v = [0]*14
# v[0]  = [5,  3.341898e-08, 0.2173221, 3.388300e+08, 1]         # x0=4*10**18,  D= 3.90970e+7
# v[1]  = [5,  3.123609e-08, 0.2172087, 3.565573e+08, 1]         # x0=e^43,      D= 4.18168e+7
# v[2]  = [4,  6.874386e-09, 0.1813398, 1.215214e+09, 1]         # x0=e^46,      D= 1.63940e+8
# v[3]  = [5,  1.208702e-09, 0.2101882, 8.262901e+09, 1]         # x0=e^50,      D= 1.06120e+9
# v[4]  = [9,  1.757070e-10, 0.2789679, 9.330703e+10, 1]         # x0=e^55,      D= 1.02884e+10
# v[5]  = [30, 4.873014e-11, 0.3832708, 9.890872e+11, 1]         # x0=e^60,      D= 7.69184e+10
# v[6]  = [82, 6.286379e-11, 0.4603978, 2.844455e+12, 1]         # x0=e^75,      D= 1.74043e+11
# v[7]  = [82, 6.270787e-11, 0.4628348, 2.361523e+12, 1]         # x0=e^90,      D= 1.84304e+11
# v[8]  = [80, 6.183604e-11, 0.4641109, 1.860117e+12, 1]         # x0=e^105,     D= 1.91886e+11
# v[9]  = [84, 6.335103e-11, 0.4660744, 2.020015e+12, 1]         # x0=e^120,     D= 1.97917e+11
# v[10] = [90, 6.590771e-11, 0.4681018, 2.198840e+12, 1]         # x0=e^135,     D= 2.02553e+11
# v[11] = [79, 6.090246e-11, 0.4666782, 2.703636e+12, 1]         # x0=e^150,     D= 2.07053e+11
# v[12] = [79, 5.965946e-11, 0.4699104, 2.695460e+12, 1]         # x0=e^300,     D= 2.30126e+11
# v[13] = [72, 5.361814e-11, 0.4699322, 1.002066e+12, 1]         # x0=e^600,     D= 2.51949e+11

# for i in range(len(v)):
#   x0 = Xo[i]
#   print('exp(', round(log(Xo[i]),4), ')', round(constraint1(v[i]),10), nstr(-Delta(v[i]),6))
# print(v)
