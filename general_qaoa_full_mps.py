#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:54:01 2020

@author: ph30n1x
"""

"""
Version II

Pure MPS with SWAP Gates on only State function.
Cost function now implemented as a network

"""

import tensornetwork as tn
from Gates import Gates as g
from SmartMPS import SmartMPS as smps
from Expectation import Expectation as exp
from SVD import SVD as svd
import numpy as np
import time

#%%
h = [2.0, -5.0, 4.5, -4.0, -1.5, -5.0, -4.5, -3.5]
h = np.array(h)

J = [[0.0, 1.0, 0.0, 0.0, 0.0, 1.5, 2.5, 1.5],
      [0.0, 0.0, 1.5, 4.5, 2.0, 1.5, 2.5, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 2.5, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.5, 2.5],
      [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 6.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 2.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
J = np.array(J)

# h = np.load('/home/ph30n1x/Chalmers/Thesis/QAOA/Q15/h.npy')
# J = np.load('/home/ph30n1x/Chalmers/Thesis/QAOA/Q15/J.npy')

# Sol = np.load('/home/ph30n1x/Chalmers/Thesis/QAOA/Q15/sol.npy')
# Sol = tn.Node(Sol)
# Sol = svd.convert_mps(Sol)
# Sol[0] = tn.Node(Sol[0].tensor.reshape(1,2,1))
# Sol[-1] = tn.Node(Sol[-1].tensor.reshape(1,2,1))
# Sol = tn.FiniteMPS(Sol)

n = len(h)

pi = np.pi
N = 100
Gamma = [0.0 + x*pi/(N-1) for x in range(N)]
Beta = [0.0 + x*pi/(N-1) for x in range(N)]

# h = [-2.0, 0.0]
# h = np.array(h)

# J = [[0.0, 2.0],
#       [0.0, 0.0]]
# J = np.array(J)

# n = len(h)

# pi = np.pi
# N = 100
# Gamma = [-0.25*pi + x*0.5*pi/(N-1) for x in range(N)]
# Beta = [-0.5*pi + x*pi/(N-1) for x in range(N)]

Cost_mps = np.zeros([N,N])
Cgs = np.zeros([N,N])


#%%

def get_Cost_mpsII(gamma_beta):
    
    Cost = 0
    Sz = tn.Node(g.get_Z())
    
    for i in range(n):
        
        g_b = tn.FiniteMPS(gamma_beta.nodes)
        g_b.apply_one_site_gate(Sz,i)
        
        C = exp.exp_MPS(g_b,gamma_beta)
        C = np.real(C)
        C = h[i]*C
        Cost = Cost + C
        
    SzSz = np.tensordot(g.get_Z(),g.get_Z(),axes = 0)
    SzSz = tn.Node(SzSz)
    
    for i in range(n-1):
                
        for j in range((n-1),i,-1):
            
            g_b = gamma_beta.nodes
            g_bcon = [tn.conj(g_b[i]) for i in range(n)]
            
            for k in range(n):
                
                if (k == i):
                    
                    g_b[k][1]^SzSz[0]
                    g_bcon[k][1]^SzSz[1]
                    
                elif (k == j):
                    
                    g_b[k][1]^SzSz[2]
                    g_bcon[k][1]^SzSz[3]
                    
                else:
                    
                    g_bcon[k][1]^g_b[k][1]
                
                if (k == (n-1)):
                    
                    g_bcon[k][2]^g_bcon[0][0]
                    g_b[k][2]^g_b[0][0]
                    
                else:
                    
                    g_bcon[k][2]^g_bcon[k+1][0]
                    g_b[k][2]^g_b[k+1][0]
                    
            
            C = tn.contractors.greedy((g_bcon + [SzSz] + g_b))
            C = C.tensor
            C = np.real(C.item())
            C = J[i,j]*C
            Cost = Cost + C
            del g_b, g_bcon
        
    
    return Cost


#%%


def QAOA_mps(gamma,beta):
    
    gamma_beta = tn.FiniteMPS(smps.plus_state(n))
    SWAP = tn.Node(g.get_SWAP())
    
    
    for i in range(n):
        
        Rz = tn.Node(g.get_Rz(2*gamma*h[i]))
        Rz = Rz.reorder_edges([Rz[1],Rz[0]])
        gamma_beta.apply_one_site_gate(Rz, i)
        # gamma_beta.position(site = 0,normalize = False)
        
    
    for i in range(n-1):
                
        for j in range((i+1),n):
            
            Jij = tn.Node(g.get_Jij(gamma, J[i][j]))
            Jij = Jij.reorder_edges([Jij[2],Jij[0],Jij[3],Jij[1]])
            
            # print('\n\n.....For i,j =',i,',',j,'.....\n\n')
            # D0 = [gamma_beta.nodes[x].get_dimension(2) for x in range(n-1)]
            # print('D0 = ',D0,'\n')
            
            for k in range(1,(j -i)):
                gamma_beta.apply_two_site_gate(SWAP, site1 = (j-k), site2 = (j-k+1))
                
            # D1 = [gamma_beta.nodes[x].get_dimension(2) for x in range(n-1)]
            # print('D1 = ',D1,'\n')
                
            gamma_beta.apply_two_site_gate(Jij, site1 = i, site2 = (i+1))
            # g_b.position(site = 0,normalize = False)
            
            # D2 = [gamma_beta.nodes[x].get_dimension(2) for x in range(n-1)]
            # print('D2 = ',D2,'\n')
                
            for k in range(1,(j -i)):
                gamma_beta.apply_two_site_gate(SWAP, site1 = (i+k), site2 = (i+k+1))
                
            # D3 = [gamma_beta.nodes[x].get_dimension(2) for x in range(n-1)]
            # print('D3 = ',D3,'\n')
            # print('..........................\n\n')
                
    
    Rx = tn.Node(g.get_Rx(2*beta))
    Rx = Rx.reorder_edges([Rx[1],Rx[0]])
    
    for i in range(n):
        
        gamma_beta.apply_one_site_gate(Rx, i)
        # gamma_beta.position(site = 0,normalize = False)
        
    
    return gamma_beta

#%%

t0 = time.time()

for i in range(N):
    
    print(i,'\n')
    for j in range(N):
        
        tj0 = time.time()
        
        gamma_beta = QAOA_mps(Gamma[i],Beta[j])
        Cost_mps[i,j] = get_Cost_mpsII(gamma_beta)
        
        # k = exp.exp_MPS(Sol,gamma_beta)
        # Cgs[i,j] = (abs(k)**2)*100
        
        del gamma_beta
        tj1 = time.time()
        print('Time for 1 iteration = ',(tj1-tj0),'s\n')

t1 = time.time()

print("\n\nTotal time elapsed between cost calc = ",(t1 - t0)/60,'mins\n\n')

#%% Plottinng

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

G,B = np.meshgrid(Beta,Gamma)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ylabel('Gamma')
plt.xlabel('Beta')
plt.title('Cost')
ax.plot_surface(G,B,Cost_mps, cmap = 'jet', rstride=1, cstride=1, linewidth=0, antialiased=False)
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.ylabel('Gamma')
# plt.xlabel('Beta')
# plt.title('Cost')
# ax.plot_surface(G,B,Cgs, cmap = 'jet', rstride=1, cstride=1, linewidth=0, antialiased=False)
# plt.show()

# del ax, G, B, fig
