#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:32:28 2022

@author: john
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sopt


"""
q[:, 0] = rho
q[:, 0] = rho*u
q[:, 0] = E

"""


def reconstruction5(q):
    """
        5th order WENO reconstruction 
    """ 
    n   = q.shape[0]
    J   = np.arange(n)
    Jm1 = np.roll(J, 1)
    Jm2 = np.roll(J, 2)
    Jp1 = np.roll(J, -1)
    Jp2 = np.roll(J, -2)
    Jp3 = np.roll(J, -3)

    ## q-
    # Polynomials
    p0n = (1/3)*q[Jm2] - (7 / 6) * q[Jm1] + (11 / 6) * q[J]
    p1n = (-1/6)*q[Jm1] + (5 / 6) * q[J] + (1 / 3) * q[Jp1]
    p2n = (1/3)*q[J] + (5 / 6) * q[Jp1] - (1 / 6) * q[Jp2]

    # Smoothness indicator
    B0n = (13 / 12) * (q[Jm2] - 2 * q[Jm1] + q[J]) ** 2 + 1 / 4 * (
        q[Jm2] - 4 * q[Jm1] + 3 * q[J]
    ) ** 2

    B1n = (13 / 12) * (q[Jm1] - 2 * q[J] + q[Jp1]) ** 2 + 1 / 4 * (
        q[Jm1] - 4 * q[J] + 3 * q[Jp1]
    ) ** 2

    B2n = (13 / 12) * (q[J] - 2 * q[Jp1] + q[Jp2]) ** 2 + 1 / 4 * (
        q[J] - 4 * q[Jp1] + 3 * q[Jp2]
    ) ** 2

    # Constants
    d0n = 1 / 10
    d1n = 6 / 10
    d2n = 3 / 10
    epsilon = 1e-6

    # alpha
    alpha0n   = d0n / (epsilon + B0n) ** 2
    alpha1n   = d1n / (epsilon + B1n) ** 2
    alpha2n   = d2n / (epsilon + B2n) ** 2
    alphasumn = alpha0n + alpha1n + alpha2n

    # Stencil weights
    w0n = alpha0n / alphasumn
    w1n = alpha1n / alphasumn
    w2n = alpha2n / alphasumn

    # q-
    qn = w0n * p0n + w1n * p1n + w2n * p2n

    ## For q+
    umm = q[Jm1]
    um  = q[J]
    u   = q[Jp1]
    up  = q[Jp2]
    upp = q[Jp3]

    # Polynomials
    p0p = (-umm + 5 * um + 2 * u)/6
    p1p = (2 * um + 5 * u - up)/6
    p2p = (11 * u - 7 * up + 2 * upp)/6

    # Smooth Indicators
    B0p = (13 / 12)*(umm - 2*um + u)**2 + 1/4*(umm - 4*um + 3*u)**2
    B1p = (13 / 12)*(um - 2*u + up)**2  + 1/4*(um - up)**2
    B2p = (13 / 12)*(u - 2*up + upp)**2 + 1/4*(3*u - 4*up + upp)**2

    # Constants
    d0p     = 3/10
    d1p     = 6/10
    d2p     = 1/10
    epsilon = 1e-6

    # Alpha weights
    alpha0p   = d0p / (epsilon + B0p) ** 2
    alpha1p   = d1p / (epsilon + B1p) ** 2
    alpha2p   = d2p / (epsilon + B2p) ** 2
    alphasump = alpha0p + alpha1p + alpha2p

    # ENO stencils weigths
    w0p = alpha0p / alphasump
    w1p = alpha1p / alphasump
    w2p = alpha2p / alphasump

    # q+
    qp = w0p * p0p + w1p * p1p + w2p * p2p

    return qp, qn


def reconstruction3(q):
    """
        3rd order WENO reconstruction 
    """    
    n   = q.shape[0]
    J   = np.arange(n)
    Jm1 = np.roll(J, 1)
    Jp1 = np.roll(J, -1)
    Jp2 = np.roll(J, -2)
    
    qp = q[Jp1]
    qm = q[Jm1]
    
    ##upwind
    #polynomials
    p0n = (-qm + 3*q)/2
    p1n = (q + qp)/2
    
    #Smoothness indicator
    B0n = (qm - q)**2
    B1n = (q - qp)**2
    
    #Constants
    d0n = 1/3
    d1n = 2/3
    epsilon = 1e-6
    
    #alpha
    alpha0n   = d0n/(epsilon + B0n)**2
    alpha1n   = d1n/(epsilon + B1n)**2
    alphasumn = alpha0n + alpha1n
    
    #Stencil weights
    w0n = alpha0n/alphasumn
    w1n = alpha1n/alphasumn
    
    un  = w0n*p0n + w1n*p1n
    
    ##Downwind
    
    qc  = q[Jp1]
    qm  = q
    qp  = q[Jp2]
    
    #Polynomials
    p0p = (qm + qc)/2
    p1p = (3*qc - qp)/2
    
    #Smooth indicators
    B0p = (qm - qc)**2
    B1p = (qc - qp)**2
    
    #Constants
    d0p = 2/3
    d1p = 1/3
    
    #alpha
    alpha0p   = d0p/(epsilon + B0p)**2
    alpha1p   = d1p/(epsilon + B1p)**2
    alphasump = alpha0p + alpha1p
    
    #Stencil weights
    w0p = alpha0p/alphasump
    w1p = alpha1p/alphasump
    
    up  = w0p*p0p + w1p*p1p
    
    return up, un
    

def EOC(Uh, U2h, Uhe, U2he, x):
    rhoh   = Uh[:, :, 0][::3, ::3]
    rho2h  = U2h[:, :, 0]
    rhohe  = Uhe[:, :, 0][::3, ::3]
    rho2he = U2he[:, :, 0]
    idx    = np.where(abs(x) <= 0.5)
    deno   = np.where(np.abs(rhoh - rhohe) < 1e-10, 0.1, np.abs(rhoh - rhohe))
    nume   = np.where(np.abs(rho2h - rho2he) < 1e-10, 0.1, np.abs(rho2h - rho2he))
    eoc    = (np.log(nume/deno)/np.log(2))[:, idx[0]]
    Eoc    = np.where(eoc == 0, 20/np.log(2), eoc)
    return Eoc
   
    
    









########### BURGERS ##########################
def f(u):
    return u**2/2

def fprime(u):
    return u

def flux(up, um, fprime, f):
    alpha = np.maximum(np.abs(fprime(um)), np.abs(fprime(up)))
    return (f(up) + f(um)) / 2 - (alpha / 2) * (up - um)


def burgers_true_sol(a, b, initial_condition, x, t):
    ic_num = initial_condition(x)
    # for safety
    min_u = np.min(ic_num) - 1
    max_u = np.max(ic_num) + 1

    u = np.zeros_like(x)
    for i, x_i in enumerate(x):

        def rootfunc(u):
            var = x_i - u * t
            var = a + (var - a) % (b - a)
            return u - initial_condition(var)

        u[i] = sopt.bisect(rootfunc, min_u, max_u, xtol=1e-10)

    return u

def initial_condition_burgers(x):
    return 0.2 + np.sin(2 * np.pi * x)



####### SODS SHOCK TUBE ##########3

def sods_true_sol(x, tf):
    gamma = 1.4
    alpha = (gamma + 1)/(gamma - 1)
    Pl    = 1.0
    Pr    = 0.1
    ul    = 0
    ur    = 0
    rhol  = 1.0
    rhor  = 0.125
    xmid  = (x[0] + x[-1])/2
    
    Prl    = Pr/Pl
    cright = np.sqrt(gamma*Pr/rhor)
    cleft  = np.sqrt(gamma*Pl/rhol)
    CRL    = cright/cleft
    Machleft = (ul - ur)/cleft
    
    def f(P):
        return (1 + Machleft*(gamma - 1)/2 - (gamma - 1)*CRL*(P - 1)
                /np.sqrt(2*gamma*(gamma - 1 + (1 + gamma)*P)))**(2*gamma/(gamma - 1))/P - Prl
    
    P34    = sopt.root_scalar(f, bracket=[2, 4], method='brentq').root
    
    P3     = P34*Pr
    rho3   = rhor*(1 + alpha*P34)/(alpha + P34)
    rho2   = rhol*(P34*Pr/Pl)**(1/gamma)
    u2     = ul - ur + (2/(gamma-1))*cleft*(1-(P34*Pr/Pl)**((gamma-1)/(2*gamma)))
    c2     = np.sqrt(gamma*P3/rho2)
    
    spos   = xmid + tf*cright*np.sqrt((gamma-1)/(2*gamma)+(gamma+1)/(2*gamma)*P34) + tf*ur
   
    conpos = xmid + u2*tf + tf*ur	# Position of contact discontinuity
    pos1   = xmid + (ul - cleft)*tf	# Start of expansion fan
    pos2   = xmid + (u2 + ur - c2)*tf	# End of expansion fan
    
    
    p      = np.zeros_like(x)
    ux     = np.zeros_like(x)
    rho    = np.zeros_like(x)
    Mach   = np.zeros_like(x)
    cexact = np.zeros_like(x)
    
    for i in range(len(x)):
        if x[i] <= pos1:
            p[i]        = Pl
            rho[i]      = rhol
            ux[i]       = ul
            cexact[i]   = np.sqrt(gamma*p[i]/rho[i])
            Mach[i]     = ux[i]/cexact[i]
            
        elif x[i] <= pos2:
            p[i]        = Pl*(1 + (pos1-x[i])/(cleft*alpha*tf))**(2*gamma/(gamma-1))
            rho[i]      = rhol*(1 + (pos1-x[i])/(cleft*alpha*tf))**(2/(gamma-1))
            ux[i]       = ul + (2/(gamma+1))*(x[i]-pos1)/tf
            cexact[i]   = np.sqrt(gamma*p[i]/rho[i])
            Mach[i]     = ux[i]/cexact[i]
            
        elif x[i] <= conpos:
            p[i]        = P3
            rho[i]      = rho2
            ux[i]       = u2 + ur
            cexact[i]   = np.sqrt(gamma*p[i]/rho[i])
            Mach[i]     = ux[i]/cexact[i]
            
        elif x[i] <= spos:
            p[i]        = P3
            rho[i]      = rho3
            ux[i]       = u2 + ur
            cexact[i]   = np.sqrt(gamma*p[i]/rho[i])
            Mach[i]     = ux[i]/cexact[i]
            
        else:
            p[i]        = Pr
            rho[i]      = rhor
            ux[i]       = ur
            cexact[i]   = np.sqrt(gamma*p[i]/rho[i])
            Mach[i]     = ux[i]/cexact[i];

    E   = p/(gamma - 1) + 0.5*rho*ux**2

    return np.array([rho, rho*ux, E]).T

def initial_condition_sod(x):
    Pl   = 1.0
    Pr   = 0.1
    ul   = 0
    ur   = 0
    rhol = 1.0
    rhor = 0.125
    xmid = (x[0] + x[-1])/2
    
    P   = np.zeros_like(x)
    rho = np.zeros_like(x)
    u   = np.zeros_like(x)
    
    P   = np.where(x > xmid, Pr, Pl)
    rho = np.where(x > xmid, rhor, rhol)
    u   = np.where(x > xmid, ur, ul)
    
    gamma = 1.4
    
    E   = P/(gamma - 1) + 0.5*rho*u**2
    
    rho_u = rho*u
    
    q = np.array([rho, rho_u, E])
    return q.T
    
   
def f_sod(q):
    gamma = 1.4
    f1 = q[:,1]
    P  = (gamma - 1)*(q[:,2] - 0.5*q[:,1]*q[:,1]/q[:,0])
    f2 = P + q[:,1]*q[:,1]/q[:,0]
    f3 = (q[:,1]/q[:,0])*(q[:,2] + P)
    
    f = np.array([f1, f2, f3])
    return f.T

def fprime_sod(q):
    u = q[:,1]/q[:,0]
    return np.array([u, u, u]).T

def flux_sod(qp, qm, fprime_sod, f_sod):
    alpha = np.maximum(np.abs(fprime_sod(qm)), np.abs(fprime_sod(qp)))
    return (f_sod(qp) + f_sod(qm)) / 2 - (alpha / 2) * (qp - qm)




##### ADVECTION

def f_adv(u):
    return u

def flux_adv(up, um, f):
    return (f(up) + f(um)) / 2 - (1 / 2) * (up - um)

def initial_condition_adv(x):
    gaussian = lambda x, height, position, hwhm: height * np.exp(-np.log(2) * ((x - position)/hwhm)**2)
    u_init = gaussian(x, 1, 0.5, 0.1)
    return u_init
    
