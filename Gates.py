#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:44:07 2020

@author: ph30n1x
"""

import numpy as np
import math
import cmath

class Gates():
    
    def get_X():
        
        x = np.array([[0.0,1.0],[1.0,0.0]])
        return x
    
    def get_Rx(theta=math.pi):
        
        if (theta == math.pi):
            Rx = np.array([[0.0,1.0],[1.0,0.0]])
            
        else:
            
            Rx = np.array([[math.cos(theta*0.5),-1.0j*math.sin(theta*0.5)],
                           [-1.0j*math.sin(theta*0.5),math.cos(theta*0.5)]], dtype = np.complex64)
            
        return Rx
            
    def get_Y():
        
        y = np.array([[0.0,-1.0j],[1.0j,0.0]], dtype = np.complex64)
        return y
                
    def get_Ry(theta=math.pi):
        
        if (theta == math.pi):
            Ry = np.array([[0.0,-1.0j],[1.0j,0.0]], dtype = np.complex64)
            
        else:
            
            Ry = np.array([[math.cos(theta*0.5),-1.0*math.sin(theta*0.5)],
                           [1.0*math.sin(theta*0.5),math.cos(theta*0.5)]], dtype = np.complex64)
            
        return Ry
    
    def get_Z():
        
        z = np.array([[1.0,0.0],[0.0,-1.0]])
        return z
            
    def get_Rz(theta=math.pi):
        
        if (theta == math.pi):
            Rz = np.array([[1.0,0.0],[0.0,-1.0]])
            
        else:
            
            Rz = np.array([[cmath.exp(-1.0j*theta*0.5),0.0],
                           [0.0,cmath.exp(1.0j*theta*0.5)]], dtype = np.complex64)
            
        return Rz
    
    def get_I():
        
        i = np.eye(2)
        return i
    
    def get_H():
        
        h = np.array([[1.0,1.0],[1.0,-1.0]])/np.sqrt(2)
        return h
    
    def get_CNOT():
        
        x = x = np.array([[0.0,1.0],[1.0,0.0]])
        i = np.array([[1.0,0.0],[0.0,1.0]])
        z = np.array([[1.0,0.0],[0.0,-1.0]])
        
        cnot = 0.5*np.tensordot((i+z),i,axes = 0)
        cnot += 0.5*np.tensordot((i-z),x,axes = 0)
        return cnot
    
    def get_CZ():
        
        i = np.array([[1.0,0.0],[0.0,1.0]])
        z = np.array([[1.0,0.0],[0.0,-1.0]])
        
        cz = 0.5*np.tensordot((i+z),i,axes = 0)
        cz += 0.5*np.tensordot((i-z),z,axes = 0)
        return cz
    
    def get_CY():
        
        y = np.array([[0.0,-1.0j],[1.0j,0.0]])
        i = np.array([[1.0,0.0],[0.0,1.0]])
        z = np.array([[1.0,0.0],[0.0,-1.0]])
        
        cy = 0.5*np.tensordot((i+z),i,axes = 0)
        cy += 0.5*np.tensordot((i-z),y,axes = 0)
        return cy
    
    def get_SWAP():
        
        i = np.array([[1.0,0.0],[0.0,1.0]])
        x = x = np.array([[0.0,1.0],[1.0,0.0]])
        y = np.array([[0.0,-1.0j],[1.0j,0.0]])
        z = np.array([[1.0,0.0],[0.0,-1.0]])
        
        swap = 0.25*np.tensordot((i + z),(i + z),axes = 0)
        swap = swap + 0.25*np.tensordot((i - z),(i - z),axes = 0)
        swap = swap + 0.25*np.tensordot((x - 1.0j*y),(x + 1.0j*y),axes = 0)
        swap = swap + 0.25*np.tensordot((x + 1.0j*y),(x - 1.0j*y),axes = 0)
        
        swap = np.real(swap)
        return swap
    
    def get_Jij(gamma,jij):
        
        z = np.array([[1.0,0.0],[0.0,-1.0]])
        i = np.array([[1.0,0.0],[0.0,1.0]])
        
        Jij = math.cos(gamma*jij)*np.tensordot(i,i,axes = 0)
        Jij = Jij.astype(np.complex64)
        Jij -= 1j*math.sin(gamma*jij)*np.tensordot(z,z,axes = 0)
        
        return Jij

        
    
