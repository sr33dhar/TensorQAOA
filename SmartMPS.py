#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:17:49 2019

@author: ph30n1x
"""

import tensornetwork as tn
import numpy as np

class SmartMPS():
    
    
    def ground_state(n,connected = False):
        
        
        MPS = [0 for i in range(n)]
        
        A_beg = tn.Node(np.array([[[1.],
                                   [0.]]]), name = "Q1", axis_names = ['a0','s1','a1'])

        if (n>1):
                        
            A_end = tn.Node(np.array([[[1.],
                                      [0.]]]),
                            name = "Q"+str(n), axis_names = ['a'+str(n-1),'s'+str(n),'a0'])
            
            MPS[0] = A_beg
            MPS[-1] = A_end
            
            
            for i in range(1,n-1):
                
                a = tn.Node(np.array([[[1.],
                                       [0.]]]), name = "Q"+str(i+1), 
                            axis_names = ['a'+str(i+1),'s'+str(i+1),'a'+str(i+2)])
                
                MPS[i] = a
            
            
        else:
            
            MPS = [A_beg]
        
        
        if(connected):
            
            for i in range(n):
                
                if (i == (n-1)):
                    
                    MPS[i][2]^MPS[0][0]
                    
                else:
                    
                    MPS[i][2]^MPS[i+1][0]
                    
        
        return MPS
    
    def plus_state(n,connected = False):
        
        MPS = [0 for i in range(n)]
        p = 1/np.sqrt(2)
        
        A_beg = tn.Node(np.array([[[p],
                                   [p]]]),name = "Q1",
                        axis_names = ['a0','s1','a1'])
        
        if (n>1):
            
                        
            A_end = tn.Node(np.array([[[p],
                                       [p]]]),name = "Q"+str(n),
                            axis_names = ['a'+str(n-1),'s'+str(n),'a0'])
            
            MPS[0] = A_beg
            MPS[-1] = A_end
            
            
            for i in range(1,n-1):
                
                a = tn.Node(np.array([[[p],
                                       [p]]]), name = "Q"+str(i+1), 
                            axis_names = ['a'+str(i+1),'s'+str(i+1),'a'+str(i+2)])
                
                MPS[i] = a
                
        
        else:
            
            MPS = [A_beg]
        
        if(connected):
            
            for i in range(n):
                
                if (i == (n-1)):
                    
                    MPS[i][2]^MPS[0][0]
                    
                else:
                    
                    MPS[i][2]^MPS[i+1][0]
                    
        
        return MPS