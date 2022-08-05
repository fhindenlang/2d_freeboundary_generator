#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Python document contains all the algorithms related to the computation
of kernels, especially the GS one.
"""
import numpy as np
import scipy.special



# interface to G
def G(r1, z1, r2, z2):
    return G_km1(r1, z1, r2, z2)

# interface to dG_dr
def dG_dr(r1, z1, r2, z2):
    return dG_dr_ex(r1, z1, r2, z2)

# interface to dG_dz
def dG_dz(r1, z1, r2, z2):
    return dG_dz_ex(r1, z1, r2, z2)


######################################

def G_km1(r1, z1, r2, z2):
    
    '''
    Compute the Kernel function G( (r,z)_1, (r,z)_2) ) of the free-space grad-shafranov operator nabla* psi = 0 (quoted in the FEM-BEM article).
    Compute 1-k^2 instead of k^2, to have a high accuracy at the pole k->1
    '''

    # Note that k² = m    
    # 0 < k < 1 is mandatory

    one_mk_sqr = ((r1 - r2) ** 2 + (z1 - z2) ** 2) / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    k_sqr = 1 - one_mk_sqr
    
    G = np.sqrt((r1+r2)**2+(z1-z2)**2)/(4*np.pi) * ((1+one_mk_sqr)*scipy.special.ellipkm1(one_mk_sqr)-2*scipy.special.ellipe(k_sqr))
    
    #k_sqr = 1 : problem

    #k_sqr = 4*r1*r2/((r1+r2)**2 + (z1-z2)**2)
        
    return G

def G_k2(r1, z1, r2, z2):
    
    "Compute the Kernel function G, directly from k^2"
    
    # 0 < k < 1 is mandatory
    
    k_sqr=  4*r1*r2 / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    G = np.sqrt((r1+r2)**2+(z1-z2)**2)/(4*np.pi) * ((2-k_sqr)*scipy.special.ellipk(k_sqr)-2*scipy.special.ellipe(k_sqr))
    
    return G

######################################



def dG_dr_FD(r1, z1, r2, z2):
    return (G(r1+1.0e-8, z1, r2, z2)-G(r1, z1, r2, z2))/1.0e-8

def dG_dr_ex(r1, z1, r2, z2): #derivative with respect to z1.
   
    one_mk_sqr = ((r1 - r2) ** 2 + (z1 - z2) ** 2) / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    k_sqr = 1 - one_mk_sqr
    
    k = np.sqrt(k_sqr)
    
    #dk/dr=1/(2k) dk^2/dr
    dk_dr=k/(2*r1)-(r1+r2)*k/((r1 + r2) ** 2 + (z1 - z2) ** 2)
    
    dG_dr = G(r1, z1, r2, z2)/(2*r1) + dk_dr * (np.sqrt((r1+r2)**2+(z1-z2)**2)/(4*np.pi*k)) * (
                      -2 *scipy.special.ellipkm1(one_mk_sqr)
                      +(2-k_sqr)/(1-k_sqr)*scipy.special.ellipe(k_sqr))
    return dG_dr


def dG_dz_FD(r1, z1, r2, z2):
    return (G(r1, z1+1.0e-8, r2, z2)-G(r1, z1, r2, z2))/1.0e-8

def dG_dz_ex(r1, z1, r2, z2): #derivative with respect to z1.
   
    one_mk_sqr = ((r1 - r2) ** 2 + (z1 - z2) ** 2) / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    k_sqr = 1 - one_mk_sqr
    
    k = np.sqrt(k_sqr)
    
    #dk/dz=1/(2k) dk^2/dz
    dk_dz=-(z1-z2)*k/((r1 + r2) ** 2 + (z1 - z2) ** 2)
    
    dG_dz = dk_dz * (np.sqrt((r1+r2)**2+(z1-z2)**2)/(4*np.pi*k)) *  (-2 *scipy.special.ellipkm1(one_mk_sqr)
                      +(2-k_sqr)/(1-k_sqr)*scipy.special.ellipe(k_sqr))
    return dG_dz

######################################

def G_smoothed(r1, z1, r2, z2):
    '''
    Computes a smoothed (but not normalized by -pi/10) version of the kernel.
    '''

    # 0 < k < 1 is mandatory
    delta = 0.0013900858621806052
    
    #one_mk_sqr = one_mm(r1, z1, r2, z2)
    one_mk_sqr = ((r1 - r2) ** 2 + (z1 - z2) ** 2) / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    one_mk_sqr = (one_mk_sqr+delta*np.exp(-one_mk_sqr/delta/2))/(1+delta*np.exp(-one_mk_sqr/delta/2))
    k_sqr = 1- one_mk_sqr
    
    G = np.sqrt((r1+r2)**2+(z1-z2)**2)/(4*np.pi) * ((1+one_mk_sqr)*scipy.special.ellipkm1(one_mk_sqr)-2*scipy.special.ellipe(k_sqr))
    #k_sqr = 1 : problem
    
    return G

def G_norm(r1, z1, r2, z2):
    "Compute the normalized Kernel function quoted in the FEM-BEM article."
    return G(r1, z1, r2, z2)/r2

def K(r1, z1, r2, z2):
    
    "Compute the first complete elliptic integral quoted in the FEM-BEM article."
    "Here, the computation of k is the old one, that causes issues when k goes to 1."

    # 0 < k < 1 is mandatory
    one_mk_sqr = ((r1 - r2) ** 2 + (z1 - z2) ** 2) / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    k_sqr = 1 - one_mk_sqr
    return scipy.special.ellipk(k_sqr)


def Km1(r1, z1, r2, z2):
    
    "Compute the first complete elliptic integral quoted in the FEM-BEM article."

    # 0 < k < 1 is mandatory
    one_mk_sqr = ((r1 - r2) ** 2 + (z1 - z2) ** 2) / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    return scipy.special.ellipkm1(one_mk_sqr)


