# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 15:59:35 2025

@author: Nicola
"""

import numpy as np 
import scipy.special 
from scipy.stats import norm
from math import pi



def TSRV(log_prices, K):

    # this is the function that returns the two steps realized volatility estimator
    # used in the paper of Corsi & Renó introduced by (vedi sotto)

    # Article : A Tale of Two Time Scales
    # Determining Integrated Volatility With Noisy High-Frequency Data
    # Authors : Lan Zhang, Per A Mykland & Yacine Aït-Sahalia
    # Year : 2005

    n = len(log_prices)

    n_bar = (n-K+1)/(K)

    c = (1 - n_bar/n)**(-1)

    avg_log_returns = log_prices[K:] - log_prices[0:n-K]

    avg_rv = np.sum(avg_log_returns**2)/K

    log_returns = log_prices[1:] - log_prices[0:-1]

    rv = np.sum(log_returns**2)

    return c*(avg_rv - (n_bar/n)*rv)


# Corrected version from Corsi, Pirino and Renò 
# Threshold bipower variation and the impact of jumps on volatility forecasting 
# Journal of econometrics 2010

def Z_gamma(x, y, c, gamma):

    if x**2<y:
        return np.abs(x)**gamma 
    
    else:

        A = 2 * norm.cdf(-c) * np.sqrt(pi) 
        B = (2/c**2 * y)**(gamma/2) 
        C = scipy.gamma(.5*(gamma+1)) * scipy.stats.gammaincc(.5*(gamma+1), .5*c**2)
        
        return 1/A * B * C  


def point_filtered_variance(xt, c=3, L=25, n_iter=3, tol=1e-6):
    
    V = np.inf 
    
    for _ in range(0, n_iter):
        
        num = 0 
        den = 0
        
        for i in range(-L, -1):
            
            x = xt[i+L]
            
            if x**2 < c**2 * V:
                
                num += norm.pdf(i/L) * x**2 
                den += norm.pdf(i/L) 
        
        for i in range(1, L):
            
            x = xt[i+L]
            
            if x**2 < c**2 * V:
                
                num += norm.pdf(i/L) * x**2 
                den += norm.pdf(i/L) 
        
        try:
            temp_V = num/den 
            
            if np.abs(V-temp_V) <= tol: 
                
                V = temp_V 
        except:
            return V 
    
    return V 


def filtered_variance(rt, c=3, L=25, n_iter=3, tol=1e-6):
    
    Vt = np.empty(len(rt))
    
    for i in range(L, len(rt)-L):
        
        xt = rt[i-L:i+L]
        
        Vt[i] = point_filtered_variance(xt, c, L, n_iter, tol)  
    
    Vt[0:L] = np.mean(Vt[L: len(rt)-L])
    Vt[len(rt)-L : len(rt)] = np.mean(Vt[L: len(rt)-L]) 
    
    return Vt
    

    
def RV(prices):
    
    return np.sum((np.log(prices[1:]) - np.log(prices[:-1]))**2) 



def CTBPV(rt, c=3, L=25, n_iter=3, tol=1e-6):
    # rt = log-returns
    mu1 = .7979 # from the paper 
    Vt = filtered_variance(rt, c, L, n_iter, tol) 
    
    result = 0
    
    for i in range(1, len(rt)):
        
        r1 = rt[i]
        r0 = rt[i-1] 
            
        result += Z_gamma(r1, c**2 * Vt[i], c, 1) * Z_gamma(r0, c**2 * Vt[i-1], c, 1)
        
    return result / mu1**2 


def CTTriPV(rt, delta, c=3, L=25, n_iter=3, tol=1e-6):

    mu3 = .8309 # from the paper 
    Vt = filtered_variance(rt, c, L, n_iter, tol) 
    
    result = 0
    
    for i in range(2, len(rt)):
        
        r3 = rt[i]
        r2 = rt[i-1] 
        r1 = rt[i-2] 
        
        result += Z_gamma(r3, c**2 * Vt[i], c, 4/3) * Z_gamma(r2, c**2 * Vt[i-1], c,4/3) * Z_gamma(r1, c**2 * Vt[i-2], c, 4/3) 
        
    return result / (mu3**3* delta)



def CTz(rv, ctbpv, cttripv, delta):
    
    num = (rv - ctbpv) / rv 
    
    A = (pi**2/4 + pi - 5)**.5
    B = max([1, cttripv/ctbpv**2])
    
    den = A * B 
    
    statistic = 1/delta**.5 * num/den 
    
    return statistic 


########### Covarianvce estimators #####################


def RCOV(log_prices1, log_prices2):
    
    # Realizded covarinace
    
    return np.sum((np.log(log_prices1[1:]) - np.log(log_prices1[:-1])) * (np.log(log_prices2[1:]) - np.log(log_prices2[:-1]))) 

    
def TSCOV(log_prices1, log_prices2, K):

    # this is the function that returns the two scale realized covariance estimator

    # Article : Estimating covariation: Epps effect, microstructure noise
    # Authors : Lan Zhang
    # Year : 2010

    n = len(log_prices1)

    n_bar = (n-K+1)/(K)

    c = (1 - n_bar/n)**(-1)

    avg_log_returns1 = log_prices1[K:] - log_prices1[0:n-K]
    avg_log_returns2 = log_prices2[K:] - log_prices2[0:n-K]

    avg_cov = np.sum(avg_log_returns1*avg_log_returns2)/K

    log_returns1 = log_prices1[1:] - log_prices1[0:-1]
    log_returns2 = log_prices2[1:] - log_prices2[0:-1]

    cov = np.sum(log_returns1*log_returns2)

    return c*(avg_cov - (n_bar/n)*cov)

