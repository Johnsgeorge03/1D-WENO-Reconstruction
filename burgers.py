#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat May  7 11:44:21 2022

@author: john
"""

from main import *


conv_nxs = [50, 50*3, 50*3**2, 50*3**3, 50*3**4]


errlo = []
errb  = []
erra  = []
hx    = []
t_burgers_before_shock = 0.1
t_burgers_after_shock  = 0.5
for n in conv_nxs:
    x   = np.linspace(0, 1, n, endpoint = False)
    u   = initial_condition_burgers(x)
    J   = np.arange(0, n)  # all vertices
    Jm1 = np.roll(J, 1)
    Jp1 = np.roll(J, -1)
    Jp2 = np.roll(Jp1, -1)
    t   = 0
    dx  = x[1] - x[0]
    hx.append(dx)
    dt  = 0.5*dx/np.max(np.abs(fprime(u)))
    Ub  = []
    Ub.append(u)
    while t < t_burgers_before_shock:
        uplus, uminus   =  reconstruction5(u)
        flx     =  flux(uplus, uminus, fprime, f)
        u_i     =  u - (dt*0.5/dx)*(flx[J] - flx[Jm1])

        uplus, uminus   =  reconstruction5(u_i)
        flx     =  flux(uplus, uminus, fprime, f)
        u       =  u - (dt/dx)*(flx[J] - flx[Jm1])
        Ub.append(u)
        t = t + dt
    
    er = np.linalg.norm(u - burgers_true_sol(0, 1, initial_condition_burgers, x, \
                                            t),ord = np.inf)

    errb.append(er)
    u   = initial_condition_burgers(x)
    Ua  = []
    Ua.append(u)
    t   = 0
    Ta  = []
    Ta.append(t)
    while t < t_burgers_after_shock:
        uplus, uminus   =  reconstruction5(u)
        flx     =  flux(uplus, uminus, fprime, f)
        u_i     =  u - (dt/dx)*(flx[J] - flx[Jm1])

        uplus, uminus   =  reconstruction5(u_i)
        flx     =  flux(uplus, uminus, fprime, f)
        u       =  0.5*(u + u_i  - (dt/dx)*(flx[J] - flx[Jm1]))
        Ua.append(u)
        t = t + dt
        
    er = np.linalg.norm(u - burgers_true_sol(0, 1, initial_condition_burgers, x, \
                                            t),ord = 1)/n
    erra.append(er)

plt.figure(figsize = (8,6))
plt.loglog(hx, errb, '-og', label = 'before_shock')
plt.loglog(hx, hx, '--', label = r'$O(\Delta x^1)$')
plt.loglog(hx, np.array(hx)**2, '--', label = r'$O(\Delta x^2)$')
plt.loglog(hx, np.array(hx)**3, '--', label = r'$O(\Delta x^3)$')
plt.loglog(hx, np.array(hx)**4, '--', label = r'$O(\Delta x^4)$')
plt.xlabel(r'$\Delta x$', fontsize = 15)
plt.ylabel(r'$||\bar{u} - u||_{L^1}$', fontsize = 15)
plt.title(r'Error vs $\Delta x$ (before shock)', fontsize = 20)
plt.legend()
plt.grid()


plt.figure(figsize = (8,6))
plt.loglog(hx, erra, '-or', label = 'after_shock')
plt.loglog(hx, hx, '--', label = r'$O(\Delta x^1)$')
plt.loglog(hx, np.array(hx)**2, '--', label = r'$O(\Delta x^2)$')
plt.loglog(hx, np.array(hx)**3, '--', label = r'$O(\Delta x^3)$')
plt.loglog(hx, np.array(hx)**4, '--', label = r'$O(\Delta x^4)$')
plt.xlabel(r'$\Delta x$', fontsize = 15)
plt.ylabel(r'$||\bar{u} - u||_{L^1}$', fontsize = 15)
plt.title(r'Error vs $\Delta x$ (after-shock)', fontsize = 20)
plt.legend()
plt.grid()
    
utrue = burgers_true_sol(0, 1, initial_condition_burgers, x, \
                                            0.00625)
