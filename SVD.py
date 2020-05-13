#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:11:19 2020

@author: ph30n1x
"""

import tensornetwork as tn
import numpy as np

class SVD():
    
    def convert_mps(state, dia = False):
        
        n = len(state.tensor.shape)
        S = [tn.Node(np.array([0])) for i in range(n)]
        
        if (dia):
                    
            D = [tn.Node(np.array([0])) for i in range(n-1)]
            
            S[0],D[0],P,_ = tn.split_node_full_svd(state,state[:1],state[1:],max_truncation_err = 10e-5)
            
            for i in range(1,n-1):
                
                S[i],D[i],P,_ = tn.split_node_full_svd(P,P[:2],P[2:],max_truncation_err = 10e-5)
                
            S[-1] = P        
            
            del state, n, P, i, dia
            
            return S, D
        
        else:
            
            S[0],D,P,_ = tn.split_node_full_svd(state,state[:1],state[1:],max_truncation_err = 10e-5)
            P = D@P
            
            for i in range(1,n-1):
                
                S[i],D,P,_ = tn.split_node_full_svd(P,P[:2],P[2:],max_truncation_err = 10e-5)
                P = D@P
                
            S[-1] = P        
            
            del state, n, P, i, dia, D
            
            return S
            
    
    def convert_mpo(ope,dia = False):
        
        n = int(len(ope.tensor.shape)*0.5)
        O = [tn.Node(np.array([0.0])) for i in range(n)]
        
        if (dia):
            
            D = [tn.Node(np.array([0.0])) for i in range(n-1)]
            
            O[0],D[0],P,_ = tn.split_node_full_svd(ope,ope[:2],ope[2:],max_truncation_err = 10e-5)
            
            for i in range(1,n-1):
                
                O[i],D[i],P,_ = tn.split_node_full_svd(P,P[:3],P[3:],max_truncation_err = 10e-5)
                
            O[-1] = P        
            
            del ope, dia, n, P, i
            
            return O, D
        
        else:
            
            O[0],D,P,_ = tn.split_node_full_svd(ope,ope[:2],ope[2:],max_truncation_err = 10e-5)
            P = D@P
            
            for i in range(1,n-1):
                
                O[i],D,P,_ = tn.split_node_full_svd(P,P[:3],P[3:],max_truncation_err = 10e-5)
                P = D@P
                
            O[-1] = P        
            
            del ope, dia, n, P, i, D
            
            return O
            
            
        