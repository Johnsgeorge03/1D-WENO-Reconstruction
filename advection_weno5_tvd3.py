#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:57:57 2022

@author: john
"""
from main import *



conv_nxs = [50, 50*3, 50*3**2, 50*3**3, 50*3**4]



errb  = []
hx    = []
for n in conv_nxs:
    x   = np.linspace(0, 1, n, endpoint = False)
    u   = initial_condition_adv(x)
    J   = np.arange(0, n)  # all vertices
    Jm1 = np.roll(J, 1)
    Jp1 = np.roll(J, -1)
    Jp2 = np.roll(Jp1, -1)
    t   = 0
    dx  = x[1] - x[0]
    hx.append(dx)
    dt  = dx
    Ub  = []
    Ub.append(u)
    nt = int( 1/ dt + 1)
    for i in range(0, nt - 1):
        uplus, uminus   =  reconstruction5(u)
        flx             =  flux_adv(uplus, uminus, f_adv)
        u_1             =  u - (dt/dx)*(flx[J] - flx[Jm1])
     
        uplus, uminus   =  reconstruction5(u_1)
        flx             =  flux_adv(uplus, uminus, f_adv)
        u_2             =  3*u/4 + u_1/4 - 0.25*(dt/dx)*(flx[J] - flx[Jm1])
        
        uplus, uminus   =  reconstruction5(u_2)
        flx             =  flux_adv(uplus, uminus, f_adv)
        u               =  u/3 + 2*u_2/3  - 2*(dt/dx)*(flx[J] - flx[Jm1])/3
        Ub.append(u)
    
    
    er = np.linalg.norm(u - initial_condition_adv(x), ord = 1)/(n)
    
    errb.append(er)
    
        

plt.figure(figsize = (8,6))
plt.loglog(hx, errb, '-og', label = 'weno5_tvdrk3')
plt.loglog(hx, hx, '--', label = r'$O(\Delta x^1)$')
plt.loglog(hx, np.array(hx)**2, '--', label = r'$O(\Delta x^2)$')
plt.loglog(hx, np.array(hx)**3, '--', label = r'$O(\Delta x^3)$')
plt.loglog(hx, np.array(hx)**4, '--', label = r'$O(\Delta x^4)$')
plt.xlabel(r'$\Delta x$', fontsize = 15)
plt.ylabel(r'$||\bar{u} - u||_{L^1}$', fontsize = 15)
plt.title(r'Error vs $\Delta x$ (advection)', fontsize = 20)
plt.legend()
plt.grid()

