#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:57:57 2022

@author: john
"""
from main import *



conv_nxs = [125, 125*2, 125*3**2, 125*3**3, 125*3**4]

errP  = []
erru  = []
errb  = []
hx    = []
t     = 0
Ub    = []
Uh    = []
U2h   = []
Uhe   = []
U2he  = []
t2h   = []
th    = []
for n in conv_nxs:
    x   = np.linspace(-1, 1, n, endpoint = False)
    u   = initial_condition_sod(x)
    J   = np.arange(0, n)  # all vertices
    Jm1 = np.roll(J, 1)
    Jp1 = np.roll(J, -1)
    Jp2 = np.roll(Jp1, -1)
    dx  = x[1] - x[0]
    hx.append(dx)
    dt  = 0.9*dx/2
    Ub  = []
    Uhe = []
    th  = []
    Ub.append(u)
    Uhe.append(u)
    
    tf  = 0.2
    t   = 0
    th.append(t)
    if n == conv_nxs[-2]:
        U2h.append(u)
        U2he.append(u)
        t2h.append(t)
    while t < tf:
        if abs(t - tf) <= 1e-14:
            break
        elif t + dt > tf:
            dt = tf - t
        uplus, uminus   =  reconstruction3(u)
        flx             =  flux_sod(uplus, uminus, fprime_sod, f_sod)
        u_1             =  u - (dt/dx)*(flx[J] - flx[Jm1])
     
        uplus, uminus   =  reconstruction3(u_1)
        flx             =  flux_sod(uplus, uminus, fprime_sod, f_sod)
        u_2             =  3*u/4 + u_1/4 - 0.25*(dt/dx)*(flx[J] - flx[Jm1])
        
        uplus, uminus   =  reconstruction3(u_2)
        flx             =  flux_sod(uplus, uminus, fprime_sod, f_sod)
        u               =  u/3 + 2*u_2/3  - 2*(dt/dx)*(flx[J] - flx[Jm1])/3
        Ub.append(u)
        t = t + dt
        Uhe.append(sods_true_sol(x, t))
        th.append(t)
        if n == conv_nxs[-2]:
            U2h.append(u)
            U2he.append(sods_true_sol(x, t))
            t2h.append(t)
    
    qtrue = sods_true_sol(x, tf)
    P     = (u[:, 2] - 0.5 * u[:, 1] ** 2 / u[:, 0]) * 0.4
    Ptrue = (qtrue[:, 2] - 0.5 * qtrue[:, 1] ** 2 / qtrue[:, 0]) * 0.4
    
    idx = np.where(abs(x) <= 0.5)
    er  = np.linalg.norm(u[:, 0][idx] - qtrue[:,0][idx], ord = 1)/(n)
    eru = np.linalg.norm((u[:, 1]/u[:, 0])[idx] \
                         - (qtrue[:,1]/qtrue[:,0])[idx], ord = 1)/(n)
    erP = np.linalg.norm(P[idx] - Ptrue[idx], ord = 1)/(n)
    errb.append(er)
    erru.append(eru)
    errP.append(erP)


eoc   = EOC(np.array(Ub), np.array(U2h), np.array(Uhe), np.array(U2he), x[::3])  

plt.figure(figsize = (8,6))
plt.imshow(eoc, origin ="lower", extent=[-0.5, 0.5, t2h[0], t2h[-1]], \
           cmap = 'gist_ncar', aspect = 'auto')
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$t$', fontsize = 15)
plt.title(r"Local convergence estimate (Sod's shock tube)", fontsize = 20)
cb = plt.colorbar()
cb.set_label("EOC")

qtrue = sods_true_sol(x, tf)
P = (Ub[-1][:, 2] - 0.5 * Ub[-1][:, 1] ** 2 / Ub[-1][:, 0]) * 0.4
Ptrue = (qtrue[:, 2] - 0.5 * qtrue[:, 1] ** 2 / qtrue[:, 0]) * 0.4

plt.figure(figsize=(8, 6))
plt.plot(x[idx], P[idx], "o", label="numerical", ms=4)
plt.plot(x[idx], Ptrue[idx], label="true")
plt.title("Pressure vs x")
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$P$', fontsize = 15)
plt.legend()
plt.grid()

plt.figure(figsize = (8,6), dpi = 800)
plt.plot(x[6700:6850], Ub[-1][:,0][6700:6850], label = "numerical", ms = 4)
plt.plot(x[6700:6850], qtrue[:,0][6700:6850], label = "true")
plt.title("Density vs x (magnified)")
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$\rho$', fontsize = 15)
plt.legend()
plt.grid()

plt.figure(figsize = (8,6), dpi = 800)
plt.plot(x[idx], Ub[-1][:,0][idx], "o", label = "numerical", ms = 4)
plt.plot(x[idx], qtrue[:,0][idx], label = "true")
plt.title("Density vs x")
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$\rho$', fontsize = 15)
plt.legend()
plt.grid()

plt.figure(figsize = (8,6))
plt.plot(x[idx], Ub[-1][:,1][idx]/Ub[-1][:,0][idx], "o", label = "numerical", ms = 4)
plt.plot(x[idx], qtrue[:,1][idx]/qtrue[:,0][idx], label="true")
plt.title("Velocity vs x")
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$u$', fontsize = 15)
plt.grid()
plt.legend()

plt.figure(figsize=(8, 6))
plt.loglog(hx, errb, "-o", label=r"$\rho$", lw=1)
plt.loglog(hx, errP, "-o", label=r"$P$", lw=1)
plt.loglog(hx, erru, "-o", label=r"$u$", lw=1)
plt.loglog(hx, hx, "--", label=r"$O(\Delta x^1)$")
plt.loglog(hx, np.array(hx) ** 2, "--", label=r"$O(\Delta x^2)$")
plt.loglog(hx, np.array(hx) ** 3, "--", label=r"$O(\Delta x^3)$")
# plt.loglog(hx, np.array(hx)**4, '--', label = r'$O(\Delta x^4)$')
plt.xlabel(r"$\Delta x$", fontsize=15)
plt.ylabel(r"$||\bar{q} - q||_{L^1}$", fontsize=15)
plt.title(r"Error vs $\Delta x$ (Sod's shock tube)", fontsize=20)
plt.legend()
plt.grid()
