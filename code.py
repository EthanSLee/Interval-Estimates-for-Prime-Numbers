#Credit to Michaela Cully-Hugill for writing this .py document.
#This is supplementary code for the computations in the paper "Explicit Interval Estimates for Prime Numbers" by M. Cully-Hugill and E. S. Lee.
#Our paper is available at https://arxiv.org/abs/2103.05986.

import sympy as sym
from sympy import N

import numpy as np
from numpy import inf, pi

import mpmath as mpm
from mpmath import fdiv, fadd, factorial, fsum, fmul, fprod, exp, log, mpf, nprint, sqrt, nstr

from scipy.optimize import minimize, differential_evolution, Bounds, brute, NonlinearConstraint
from scipy.integrate import quad

from math import comb, prod, floor
from decimal import Decimal, getcontext
from fractions import Fraction

mpm.mp.dps = 300

def Delta(x):
    m=floor(x[0])
    d=mpf(x[1])
    a=x[2]
    result = -1/(1 - 1/(exp(d/m)*(1 + d)) - (d*a)/(exp(d/m)*(1 + d))) #maximise Delta by minimising -Delta
    return result

def X0(m, d, a, x0):
    return x0/(exp(d/m)*(1 + d))
    
def An(n):
    return Fraction((n + 1)**(n + 1)/n**n)

def Bn(n):
    return n/(n + 1)

def fma(m,n,a):
    list1 = []
    for b in range(m+1):
        list1.append((a**(m*n + 1 + b)*(1 - a)**(m-b))/factorial(m*n + b + 1))
    return An(n)**m * factorial(m*n) * fsum(list1)
    # for b in range(m+1):
    #     list1.append((a**m*(1 - a)**(m*n + 1)*((1 - a)/a)**b - a**(m*n + 1)*(1 - a)**m*(a/(1 - a))**b)/(factorial(m - b)*factorial(m*n + b + 1)))
    # return An(n)**m * factorial(m)*factorial(m*n) * fsum(list1)

def fm1(m,n):
    list1 = range(m*n + 1, m*n + m + 1 + 1)
    result1 = fprod(list1)
    return An(n)**m * factorial(m)/result1  

def vf1(m, n, a):
    return fma(m, n, a)/fm1(m, n)

def L1(m, n, d):
    list1 = range(m*n + 1, m*n + m + 1 + 1)
    prod1 = fprod(list1)
    return (1 + d)**2*2*(Bn(n)**n - Bn(n)**(n + 1))**m * prod1/factorial(m)

def prod0(m,n,k):
    list1 = range(m*n + k + 1, 2*m*n + k - m + 1 + 1) #additional 1 for range
    return prod(list1)

def L(m,n,d):
    list1 = range(m*n + 1, m*n + m + 1 + 1)
    prod1 = fprod(list1)
    list2 = []
    for k in range(m+1):
        list2.append(((-1)**(m*n + m + k))*comb(m, k)/prod0(m,n,k))
    return (1 + d)**(m + 3/2)*prod1/factorial(m)*sqrt(factorial(m*n + m)*fsum(list2))

def ZF(T):
    return 1/(R0*log(T))
def R(T):
    return a1*log(T) + a2*log(log(H)) + a3 # approximation T=H
def P(T):
    return T/(2*pi)*log(T/(2*pi)) - T/(2*pi) + 7/8 + R(T)
def q(T):
    return (a1*log(T) + a2)/(T*log(T)*log(T/(2*pi)))

def ZD(s, T):
    return A*log(T)**(5 - 2*s)*T**(8/3*(1 - s)) + B*(log(T))**2
def ZDderv(s, T):
    return (5 - 2*s)*A*log(T)**(4 - 2*s)*T**(8/3*(1 - s)-1) + A*log(T)**(5 - 2*s)*(8/3*(1 - s))*T**(8/3*(1 - s)-1) + 2*B*log(T)/T

def S1(T):
    return (1/(2*pi) + q(T0))*(log(T/T0)*log(sqrt(T*T0)/(2*pi))) + 2*R(T0)/T0 #fixed KL: * -> +
def S2(m, T):
    return (1/(2*pi) + q(T))*((1 + m*log(T/(2*pi)))/(m**2*T**m) - (1 + m*log(H/(2*pi)))/(m**2*H**m)) + 2*R(T)/T**(m + 1)
def S3(m):
    return (1/(2*pi) + q(H))*(1 + m*log(H/(2*pi)))/(m**2*H**m) + 2*R(H)/H**(m + 1)
def S4(m,s):
    dZD = lambda t: ((5 - 2*s)*A*log(t)**(4 - 2*s)*t**(8/3*(1 - s)-1) + A*log(t)**(5 - 2*s)*(8/3*(1 - s))*t**(8/3*(1 - s)-1) + 2*B*log(t)/t)/t**(m + 1)
    #return 2*ZD(s, H)/H**(m + 1) #simplified expression for optimising with higher Riemann height
    return ZD(s, H)/H**(m + 1) + sum(list(quad(dZD, H, inf)))
def S5(X,m,s):
    dZD = lambda t: ((5 - 2*s)*A*log(t)**(4 - 2*s)*t**(8/3*(1 - s)-1) + A*log(t)**(5 - 2*s)*(8/3*(1 - s))*t**(8/3*(1 - s)-1) + 2*B*log(t)/t)/t**(m + 1)
    #return 2*X**(-1/(R0*log(H)))*ZD(s, H)/H**(m + 1) #simplified expression for optimising with higher Riemann height
    return X**(-1/(R0*log(H)))*ZD(s, H)/H**(m + 1) + sum(list(quad(dZD, H, inf)))

def F0mUpper(d):
    return 1 + d
def F1mUpper(m,n,d):
    return L1(m,n,d)
def FmmUpper(m,n,d):
    return L(m,n,d)

def Sum01(m,n,d):
    return 4*F1mUpper(m,n,d)/((exp(d/(2*m)) + 1)*d)*S0
def Sum02(m,n,d):
    return 4*F0mUpper(d)/(exp(d/(2*m)) + 1)*N0
def Sum11(m,n,d,T):
    return 4*F1mUpper(m,n,d)/((exp(d/(2*m)) + 1)*d)*S1(T)
def Sum12(m,n,d,T):
    return 4*F0mUpper(d)/(exp(d/(2*m)) + 1)*(P(T) + R(T) - N0)

def B0(m,n,d):
    return min(Sum01(m,n,d),Sum02(m,n,d))
def B1(m,n,d,T):
    return min(Sum11(m,n,d,T), Sum12(m,n,d,T))
def B2(m,n,d,T):
    return fdiv(2*FmmUpper(m,n,d)*S2(m, T),fmul((exp(d/(2*m)) - 1),d**m))
def B3(m,n,d,s):
    return fdiv(2*FmmUpper(m,n,d)*S3(m)*fdiv((exp(d/m*s) + 1),(exp(d/m) - 1)),d**m)
def B41(X,m,n,d,s):
    return fdiv(2*FmmUpper(m,n,d)*S5(X, m, s)*fdiv((exp(d/m) + 1),(exp(d/m) - 1)),d**m)
def B42(m,n,d,s):
    return fdiv(2*FmmUpper(m,n,d)*S4(m, s)*fdiv((exp(d/m) + 1),(exp(d/m) - 1)),d**m)

def Sum0(m,n,d,a,x0):
    return B0(m,n,d)*X0(m, d, a, x0)**(-1/2)
def Sum1(m,n,d,a,T,x0):
    return B1(m,n,d,T)*X0(m, d, a, x0)**(-1/2)
def Sum2(m,n,d,a,T,x0):
    return B2(m,n,d,T)*X0(m, d, a, x0)**(-1/2)
def Sum3(m,n,d,a,s,x0):
    return B3(m,n,d,s)*X0(m, d, a, x0)**(s - 1) + B3(m,n,d, 1 - s)*X0(m, d, a, x0)**(-s)
def Sum4(m,n,d,a,s,x0):
    return B41(X0(m, d, a, x0), m,n, d, s)*X0(m, d, a, x0)**(-ZF(H)) + B42(m,n, d, s)*X0(m, d, a, x0)**(-1 + ZF(H)) 

def TheSum(m,n,d,a,T,s,x0):
    result = Sum0(m, n, d, a, x0) + Sum1(m, n, d, a, T, x0) + Sum2(m, n, d, a, T, x0) + Sum3(m, n, d, a, s, x0) + Sum4(m, n, d, a, s, x0)
    #print(result)
    return result

def BrunTerms(m,n,d,a,x0):
    result = (d/m)/(2*(exp(d/m) - 1))*X0(m, d, a, x0)**(-2) + w/(exp(d/m) - 1)*X0(m, d, a, x0)**(-1/2) + 2*vf1(m,n,a)*(1 + d*a)*(d/m + log(X0(m, d, a, x0)*(1 + d)))/log((exp(d/m) - 1)*X0(m, d, a, x0))
    #print(round(result,1))
    return result

def constraint1(x): #change to constraint1(x,X) and comment out fixed X
    m=floor(x[0])
    d=mpf(x[1])
    a=x[2]
    T=x[3]
    n=floor(x[4])
    s=s0
    x0=X
    # return value is f(x) in the condition f(x)\geq 0
    return 1 - TheSum(m, n, d, a, T, s, x0) - BrunTerms(m, n, d, a, x0)

R0 = 5.573412
H = 3000175332800 #Riemann height
T0 = 104537615
N0 = 2.6*10**8
S0 = 21.98308

w = 1.0344*10**(-3)
a1 = 0.11
a2 = 0.29
a3 = 2.29
A  = 58773/10000
B  = 3869/1000
s0 = Fraction(7804/10000)   #sigma_0

X = exp(90)

##------------To compute the constraint and Delta for a specific X----------------
# j = [2, 5.488475*10**(-264), 0.98612108, 2.317168*10**12, 8701]
# print(round(constraint1(j),6), nstr(Delta(j),5))

#-------------Differential evolution method-------------------
# Uncomment this section to run the optimiser
# Note the dps for mpmath may need to be increased for larger X - can use 100 for smallest X, and 400 for largest X
nlc = NonlinearConstraint(constraint1, 0, inf)  #constraint1 must be greater than 0
bounds = [(2,25), (10**(-25),10**(-20)), (0.9,0.999999), (T0,10**10), (505,505)] #bounds on parameters m, delta, a, T, n

Nfeval = 1
def callbackF(Xi, convergence):
    global Nfeval
    print((Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], round(constraint1(Xi),5), nstr(Delta(Xi),5)))
    Nfeval += 1

result = differential_evolution(Delta, bounds, constraints=(nlc), seed=1, disp=True, callback=callbackF, polish=False)
print(result.x, result.fun)

##--------------Returned parameter values-------------
## Uncomment this section to compute the constraint and Delta for each set of optimised parameter values
## Also need to change definition of constraint1, and use sufficienty high dps for mpmath (e.g. 350)

# Xo=[4*10**18, exp(43), exp(46), exp(50), exp(55), exp(60), exp(75), exp(90), exp(105), exp(120), exp(135), exp(150), exp(300), exp(600), exp(1200)]

# v = [0]*15
# v[0]  = [14, 9.787353e-12, 0.9999637, 1.045720e+08, 80]           # x0=4*10**18,  D= 1.42969e+12
# v[1]  = [15, 9.389145e-12, 0.9999975, 1.048194e+08, 85]           # x0=e^43,      D= 1.59753e+12
# v[2]  = [8,  8.974125e-13, 0.9999805, 1.048684e+08, 93]           # x0=e^46,      D= 8.91313e+12
# v[3]  = [14, 2.032975e-13, 0.9999986, 1.458169e+08, 131]          # x0=e^50,      D= 6.88633e+13
# v[4]  = [8,  9.448791e-15, 0.9998918, 2.024492e+08, 158]          # x0=e^55,      D= 8.45937e+14
# v[5]  = [12, 1.161970e-15, 0.9999607, 4.929877e+08, 205]          # x0=e^60,      D= 1.03224e+16
# v[6]  = [16, 8.571120e-19, 0.9999711, 1.749597e+11, 309]          # x0=e^75,      D= 1.86587e+19
# v[7]  = [7,  2.075109e-22, 0.9999275, 1.969037e+12, 505]          # x0=e^90,      D= 3.37161e+22
# v[8]  = [16, 2.620285e-25, 0.9999360, 2.449564e+12, 653]          # x0=e^105,     D= 6.09997e+25
# v[9]  = [16, 1.449833e-28, 0.9999240, 1.220336e+12, 962]          # x0=e^120,     D= 1.10224e+29
# v[10] = [7,  3.510751e-32, 0.9999997, 1.056601e+12, 1123]         # x0=e^135,     D= 1.99387e+32 
# v[11] = [3,  8.317890e-36, 0.9965160, 2.023982e+11, 1457]         # x0=e^150,     D= 3.56938e+35
# v[12] = [12, 8.906557e-68, 0.9999987, 1.155261e+12, 2173]         # x0=e^300,     D= 1.34730e+68
# v[13] = [13, 1.408453e-132, 0.9928829, 1.179372e+12, 4895]        # x0=e^600,     D= 8.44833e+132
# v[14] = [7,  1.919409e-263, 0.9996089, 1.193274e+12, 8710]        # x0=e^1200,    D= 3.63700e+263

# for i in range(15):
#   print(round(log(Xo[i]),4), round(constraint1(v[i],Xo[i]),10), nstr(-Delta(v[i]),6))

## ----------- Optimised parameter values with higher Riemann heights----------------
## To run the optimiser with higher Riemann heights, change return for S4 and S5, otherwise there is an overflow error

# For x0=4*10**18,
# H = 10^20: [8 5.5953448e-12 0.99998753 1.04618060e+08 83] D= 1.42961735e+12
# H = 10^30: [8 5.5943811e-12 0.99997880 1.04622682e+08 77] D= 1.429763784e+12
# H = 10^50: [3 2.1323692e-12 0.95903059 1.04631925e+08 94] D= 1.252894736e+12
# For x0=exp(90)
# H = 10^20: [15 4.4431125e-22 0.99999706 7.198698e+09 505] D = 3.37586305e+22
# H = 10^40: [15 4.4431125e-22 0.99999706 7.198698e+09 505] D = 3.375863059e+22
