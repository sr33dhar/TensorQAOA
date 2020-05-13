#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:22:49 2020

@author: ph30n1x
"""

import tensornetwork as tn

class Expectation():
    
    def exp_Node(state1,state2):
                
        n = len(state1.get_all_dangling())
        
        state1 = tn.conj(state1)
        
        if (n != len(state2.get_all_dangling())):
            
            print("\n\nError!! The number of dangling edges do not match!\n")
            print("length of state 1 is ",n,' whereas the length of state 2 is ',len(state2.get_all_dangling()))
            return None
            
        else:
            
            for i in range(n):
                
                state1[i]^state2[i]
                
            exp = (state1@state2).tensor
            exp = exp.item()
           
        return exp
    
    
    def exp_MPS(state1,state2):
        
        S1 = state1.nodes
        S2 = state2.nodes
        n = len(S1)
        
        if (n != len(S2)):
            
            print("\n\nError!! The number of dangling edges do not match!\n")
            print("length of state 1 is ",n,' whereas the length of state 2 is ',len(S2))
            return None
            
        else:
            
            S1 = [tn.conj(S1[i]) for i in range(n)]
            
            for i in range(n):
                
                S1[i][1]^S2[i][1]
                
                if (i == (n-1)):
                    
                    S1[i][2]^S1[0][0]
                    S2[i][2]^S2[0][0]
                    
                else:
                    
                    S1[i][2]^S1[i+1][0]
                    S2[i][2]^S2[i+1][0]
                    
            
            exp = tn.contractors.greedy((S1 + S2))
            
            exp = exp.tensor
            exp = exp.item()
           
        return exp        
        
        
        
 
