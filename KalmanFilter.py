# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:41:38 2025

@author: Nicola
"""

import numpy as np
from scipy.optimize import minimize
from copy import deepcopy, copy
from numpy.linalg import inv
from scipy.special import factorial, gamma
from numba import njit
from math import pi

@njit
def my_factorial(n):
    
    if n==0 or n==1:
        return 1
    
    result = 1
    
    i = 0
    
    while n-i>1:
        
        result = result * (n-i)
        
        i+=1
    return result


@njit
def sanitize_square(mat, d, jitter=1e-12):
    for i in range(d):
        for j in range(d):
            if not np.isfinite(mat[i, j]):
                mat[i, j] = 0.0
    for i in range(d):
        mat[i, i] += jitter
    # Symmetrize
    for i in range(d):
        for j in range(i + 1, d):
            avg = 0.5 * (mat[i, j] + mat[j, i])
            mat[i, j] = avg
            mat[j, i] = avg


def multivariate_gaussian_log_likelihood(data, mean, covariance):
    # n, d = data.shape
    n = len(data)
    diff = data - mean

    # Invert the covariance matrix
    inv_covariance = np.linalg.inv(covariance)
    
    log_likelihood = 0

    # Calculate the log-likelihood
    
    log_likelihood = -0.5 * (n * np.log(2 * np.pi) + np.log(np.abs(np.linalg.det(covariance))) +
                             diff @ inv_covariance @ np.transpose(diff))
    
    return log_likelihood

@njit
def Fast_multivariate_gaussian_log_likelihood(x, mean, covariance):
    diff = x - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return -np.inf

    sol = np.linalg.solve(covariance, diff)
    quad_form = diff @ sol
    
    d = len(x)
    log_likelihood = -0.5 * (d * np.log(2 * np.pi) + logdet + quad_form)
    return log_likelihood

@njit 
def Fast_multivariate_gaussian_log_likelihood_approx(x, mean, covariance, n=40):
    # this formula uses polynomial approximation (Taylor around zero) for the log
    # It is usefull when the variance covariance matrix is small
    diff = x-mean 
    sol = np.linalg.solve(covariance, diff)
    quad_form = diff @ sol 
    
    a = np.abs(np.linalg.det(covariance)) - 1
    
    result = 0
    
    for i in range(1, n+1):
        
        result += (-1)**(i+1) * a**i / i 
    
    result = result *.5
    
    result += quad_form
    
    return result

def sigmoid_like_fun(x):
    
    return 2 / (1 + np.exp(-x)) - 1

def inv_sigmoid_like_fun(y):
    
    if abs(y)>1:
        raise ValueError('input y has to be in [-1,1]')
    
    return np.log((1+y)/(1-y))

def matrix_generator(array, n_var):
    
    sigmas = array[0:n_var]
    sigmas = np.reshape(sigmas, (1, n_var))
    S = sigmas.T @ sigmas
    
    rhos = array[n_var:]
    # print(rhos)
    corr_matrix = np.zeros((n_var, n_var))
    
    index = 0
    
    for i in range(0, n_var):
        for j in range(i+1, n_var):
            
            rho = rhos[index]
            # print(rho)
            corr_matrix[i,j] = rho
            
            index += 1 
    
    return (corr_matrix + corr_matrix.T) * S + np.diag(np.diag(S))

def LeadLag_initial_params(n):
    params = np.ones(n*n) # Initial guess 
    params = np.hstack((params, np.zeros(n*2)))
    params = np.hstack((params, np.zeros( int( gamma(n+1)/(gamma(n-1)*2))))) 

    return params

def LeadLagKF(Y, Z_t, T_t, Ht, Qt, burnIn=0):
    
    a0 = np.hstack((Y[0,:], Y[0,:]))
    
    nan_index = np.isnan(a0)
    
    if True in nan_index:
        
        a0[nan_index]=0
    
    P0 = np.eye(len(Qt))
    
    at = np.empty((len(Y[:,0]), len(a0))) # Conditional mean of the unobserved process
    # Pt = [P0] # List of conditional variance of the unobserved process
    
    N = len(Y)
    M = np.shape(P0)[0] 
    K = np.shape(P0)[1]
    
    Pt = np.empty((N, M, K))
    Pt[0] = P0
    
    att = np.empty((len(Y[:,0]), len(a0) )) # filtered estimator of the unobserved process
    
    at[0] = a0 # initial value for "at"
    
    vt = np.empty((len(Y[:,0]), len(Y[0]))) # innovation error
    vt0 = np.hstack((Y[0,:] )) - a0[:-int(len(a0)/2)] # initial value of the innovation error
    
    nan_index = np.isnan(vt0)
    
    if True in nan_index:
        
        vt0[nan_index]=0
    
    vt[0] = vt0
    
    F0 = Z_t@P0@np.transpose(Z_t) + Ht
    # Ft = [] # list of variance of the innovation error
    # Ft.append(Z_t@P0@np.transpose(Z_t) + Ht) # the initial value of the first innovation error
    N = len(Y)
    M = np.shape(F0)[0] 
    K = np.shape(F0)[1]
    
    Ft = np.empty((N, M, K))
    Ft[0] = F0
    
    # Ptt = [] # filtered estimator of the variance of the unobserved process
    Ptt0 = P0 - P0@np.transpose(Z_t)@np.linalg.inv(Ft[0])@Z_t@P0
    Ptt = np.empty((len(Y), len(Ptt0), len(Ptt0[0]) ))
    Ptt[0] = Ptt0
    # Ptt.append(P0- P0@np.transpose(Z_t)@np.linalg.inv(Ft[0])@Z_t@P0) # computing the initial value of the filtered estimator of the variance of the unobserved process
    
    att[0,:] = at[0] + P0@np.transpose(Z_t)@np.linalg.inv(Ft[0])@vt[0] # computing the initial value of the filtered estimator of the unobserved process
    
    K0 = T_t@Pt[0]@np.transpose(Z_t)@np.linalg.inv(Ft[0])
    
    N = len(Y)
    M = np.shape(K0)[0] 
    K = np.shape(K0)[1]
    
    Kt = np.empty((N, M, K))
    Kt[0] = K0
    
    for i in np.arange(1, len(Y[:,0])):
        
        nan_index = np.isnan(Y[i])
        
        if True in nan_index:
            
            y = deepcopy(Y[i])
            y[nan_index] = 0
            
            Z = deepcopy(Z_t)
            
            for j in range(0, len(Z_t)):
                
                Z[nan_index] = 0
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3) and i>2:
                    Pt[i] = copy(Pt[i-1])
                    # Ft[i] = A @ Pt[i] @ A.T + H
                else:
                    Pt[i] = (T_t@Ptt[i-1]@np.transpose(T_t) + Qt) # Computing the variance
                    # Ft[i] = (Z @ Pt[i] @ Z.T + H) # computing the variance of the innovation error
            except:
                pass
            
            # Pt[i] = (T_t@Ptt[i-1]@np.transpose(T_t) + Qt) # Computing the innovation variance
            
            auxH = np.eye(len(Ht))
            auxH[~nan_index] = Ht[~nan_index]
            
            Ft[i] = (Z@Pt[i]@np.transpose(Z) + auxH) # computing the variance of the innovation error
            # Ft[i] = (Z@Pt[i]@np.transpose(Z) + Ht) # computing the variance of the innovation error
            try:
                np.linalg.inv(Ft[i])
            except:
                Ft[i] = Ft[i-1]
            
            Ptt[i] = (Pt[i] - Pt[i]@np.transpose(Z)@np.linalg.inv(Ft[i])@Z@Pt[i]) # computing the filtered variance
            
            at[i] = T_t@att[i-1]
            vt[i] = y - Z@at[i]
            
            att[i] = (at[i] + Pt[i]@np.transpose(Z)@np.linalg.inv(Ft[i])@np.transpose(vt[i]))
            
            Kt[i] = T_t@Pt[i]@np.transpose(Z_t)@np.linalg.inv(Ft[i]) # Kalman gain
            
        else:
            
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3) and i>2:
                    Pt[i] = copy(Pt[i-1])
                    # Ft[i] = A @ Pt[i] @ A.T + H
                else:
                    Pt[i] = (T_t@Ptt[i-1]@np.transpose(T_t) + Qt) # Computing the variance
                    # Ft[i] = (Z @ Pt[i] @ Z.T + H) # computing the variance of the innovation error
            except:
                pass
        
            # Pt[i] = (T_t@Ptt[i-1]@np.transpose(T_t) + Qt) # Computing the variance
            
            Ft[i] = (Z_t@Pt[i]@np.transpose(Z_t) + Ht) # computing the variance of the innovation error
            try:
                np.linalg.inv(Ft[i])
            except:
                Ft[i] = Ft[i-1]
                
            Ptt[i] = (Pt[i] - Pt[i]@np.transpose(Z_t)@np.linalg.inv(Ft[i])@Z_t@Pt[i]) # computing the filtered variance
            
            at[i] = T_t@att[i-1]
            vt[i] = Y[i] - Z_t@at[i]
            
            att[i] = (at[i] + Pt[i]@np.transpose(Z_t)@np.linalg.inv(Ft[i])@np.transpose(vt[i]))
            
            Kt[i] = T_t@Pt[i]@np.transpose(Z_t)@np.linalg.inv(Ft[i]) # Kalman gain

    loglike = 0
    n_var = len(Y[0])
    
    for i in range(burnIn, len(vt)):
        
        loglike += multivariate_gaussian_log_likelihood(vt[i], np.zeros(n_var), Ft[i])
    
    return att, Ptt, at, Pt, vt, Ft, Kt, loglike 


# def LeadLagKF2(y, Q, R, T_t, Z_t, jitter=1e-8, verbose=False):
    
#     """
#     INPUTS:
#       y : (T, d) array, observed data (NaN for missing)
#       Q : (2d, 2d) innovation covariance
#       R : (d, d) observation covariance
#       F : (2d, 2d) state transition matrix
#       C : (d, 2d) observation matrix
#       jitter : float, small diagonal regularization for singular F_t
#       verbose : bool, print regularization warnings

#     OUTPUTS:
#       at  : predicted states  (T, 2d)
#       att : filtered states   (T, 2d)
#       Pt  : predicted covariances (T, 2d, 2d)
#       Ptt : filtered covariances  (T, 2d, 2d)
#       vt  : innovations (T, d)
#       Ft  : innovation covariances (T, d, d)
#       loglik : per-step log-likelihoods
#     """
#     F = T_t
#     C = Z_t
    
#     T, d = y.shape
#     dx = 2 * d
#     Id = np.eye(d)

#     # Allocate arrays
#     at = np.zeros((T, dx))
#     att = np.zeros((T, dx))
#     Pt = np.zeros((T, dx, dx))
#     Ptt = np.zeros((T, dx, dx))
#     vt = np.zeros((T, d))
#     Ft = np.zeros((T, d, d))
#     Kt = np.zeros((T, dx, d))
#     loglik = np.zeros(T)

#     # Handle missing data mask
#     a = ~np.isnan(y)
#     y_filled = np.where(np.isnan(y), 0, y)

#     # Initialization (diffuse)
#     at[0, :] = np.tile(y_filled[0, :], 2)
#     Pt[0, :, :] = np.eye(dx)

#     # Main filtering loop
#     for t in range(T):
#         # --- D matrix for missing-data handling ---
#         D = np.vstack([
#             Id[a[t, :], :],
#             np.zeros((np.count_nonzero(~a[t, :]), d))
#         ])

#         auxY = D @ y_filled[t, :]
#         auxC = D @ C
#         auxR = D @ R @ D.T
#         diag_auxR = np.diag(auxR)
#         auxR[np.where(diag_auxR == 0), np.where(diag_auxR == 0)] = 1.0

#         # --- Prediction step ---
#         if t > 0:
#             at[t, :] = F @ att[t - 1, :]
#             Pt[t, :, :] = F @ Ptt[t - 1, :, :] @ F.T + Q

#         # --- Innovation ---
#         F_t = auxC @ Pt[t, :, :] @ auxC.T + auxR
#         F_t = (F_t + F_t.T) / 2  # symmetrize

#         cond_F = np.linalg.cond(F_t)
#         if np.isnan(cond_F) or cond_F > 1e12:
#             if verbose:
#                 print(f"Regularizing F_t at t={t}, cond={cond_F:.2e}")
#             F_t += np.eye(F_t.shape[0]) * jitter

#         # --- Kalman gain (stable solve) ---
#         try:
#             K_t = np.linalg.solve(F_t.T, (Pt[t, :, :] @ auxC.T).T).T
#         except np.linalg.LinAlgError:
#             if verbose:
#                 print(f"Singular F_t at t={t}, adding jitter {jitter}")
#             F_t += np.eye(F_t.shape[0]) * jitter
#             K_t = np.linalg.solve(F_t.T, (Pt[t, :, :] @ auxC.T).T).T

#         Kt[t, :, :auxC.shape[0]] = K_t

#         # --- Update step ---
#         v_t = auxY - auxC @ at[t, :]
#         att[t, :] = at[t, :] + K_t @ v_t
#         Ptt[t, :, :] = Pt[t, :, :] - K_t @ auxC @ Pt[t, :, :]
#         Ptt[t, :, :] = (Ptt[t, :, :] + Ptt[t, :, :].T) / 2  # enforce symmetry

#         vt[t, :len(v_t)] = v_t
#         Ft[t, :auxC.shape[0], :auxC.shape[0]] = F_t

#         # --- Log-likelihood ---
#         try:
#             sign, logdet = np.linalg.slogdet(F_t)
#             loglik[t] = -0.5 * (logdet + v_t.T @ np.linalg.solve(F_t, v_t))
#         except np.linalg.LinAlgError:
#             loglik[t] = np.nan

#     return at, att, Pt, Ptt, vt, Ft, Kt, np.sum(loglik)


# Filtering function for the Lead-Lag model optimized for numba
@njit
def FastLeadLagKF(Y, Z_t, T_t, Ht, Qt, burnIn=0, Lasso=False, lambda_=.5, Ridge=False, lambda2_=.5):
    # Ensure everything is float64
    Y   = Y.astype(np.float64)
    Z_t = Z_t.astype(np.float64)
    T_t = T_t.astype(np.float64)
    Ht  = Ht.astype(np.float64)
    Qt  = Qt.astype(np.float64)
    
    n_obs, n_y = Y.shape
    n_state = 2 * n_y  # because a0 = [Y[0], Y[0]]
    
    # Initial state
    a0 = np.empty(n_state, dtype=np.float64)
    for j in range(n_y):
        if np.isnan(Y[0,j]):
            val = 0.0
        else:
            val = Y[0,j]
        a0[j] = val
        a0[j + n_y] = val

    # Allocate arrays
    at  = np.empty((n_obs, n_state), dtype=np.float64)
    att = np.empty((n_obs, n_state), dtype=np.float64)
    Pt  = np.empty((n_obs, n_state, n_state), dtype=np.float64)
    Ptt = np.empty((n_obs, n_state, n_state), dtype=np.float64)
    vt  = np.empty((n_obs, n_y), dtype=np.float64)
    Ft  = np.empty((n_obs, n_y, n_y), dtype=np.float64)
    Kt  = np.empty((n_obs, n_state, n_y), dtype=np.float64)
    
    # Initial values
    Pt[0]  = np.eye(n_state, dtype=np.float64)
    at[0]  = a0
    vt0    = np.empty(n_y, dtype=np.float64)
    for j in range(n_y):
        vt0[j] = Y[0,j] if not np.isnan(Y[0,j]) else 0.0
    vt[0]  = vt0
    
    Ft[0]  = Z_t @ Pt[0] @ Z_t.T + Ht
    Ptt[0] = Pt[0] - Pt[0] @ Z_t.T @ np.linalg.solve(Ft[0], Z_t @ Pt[0])
    att[0] = at[0] + Pt[0] @ Z_t.T @ np.linalg.solve(Ft[0], vt[0])
    
    tmp    = T_t @ Pt[0] @ Z_t.T
    Kt[0]  = np.linalg.solve(Ft[0].T, tmp.T).T
    
    # === Filtering loop ===
    for i in range(1, n_obs):
        # Predict state covariance
        Pt[i] = T_t @ Ptt[i-1] @ T_t.T + Qt
        
        # Predict innovation covariance
        Ft[i] = Z_t @ Pt[i] @ Z_t.T + Ht
        # if np.linalg.cond(Ft[i]) > 1e12:
        #     Ft[i] = Ft[i-1]
        
        det_Ft = np.linalg.det(Ft[i])
        if np.abs(det_Ft) < 1e-12:
            Ft[i] = Ft[i-1]
        
        # Filtered covariance
        ZP = Z_t @ Pt[i]
        Ptt[i] = Pt[i] - (Pt[i] @ Z_t.T) @ np.linalg.solve(Ft[i], ZP)
        
        # State prediction
        at[i] = T_t @ att[i-1]
        
        # Innovation
        for j in range(n_y):
            if np.isnan(Y[i,j]):
                vt[i,j] = 0.0
            else:
                vt[i,j] = Y[i,j] - (Z_t @ at[i])[j]
        
        # Update state
        att[i] = at[i] + Pt[i] @ Z_t.T @ np.linalg.solve(Ft[i], vt[i])
        
        # Kalman gain
        tmp = T_t @ Pt[i] @ Z_t.T
        Kt[i] = np.linalg.solve(Ft[i].T, tmp.T).T
    
    # Log-likelihood
    loglike = 0.0
    mean_zero = np.zeros(n_y, dtype=np.float64)
    
    for i in range(burnIn, n_obs):
        loglike += Fast_multivariate_gaussian_log_likelihood(vt[i], mean_zero, Ft[i])
        
        if Lasso:
            penalty = lambda_ * np.sum(np.abs((T_t[:n_y,:n_y]-np.eye(n_y))).flatten())   # penalizing by thelead-lag parameters
            loglike = loglike - penalty
            
        # elif Lasso and Full:
        #     penalty += lambda_ * np.sum(np.abs(Ht.flatten()))
        #     penalty += lambda_ * np.sum(np.abs(Qt[:n_y,:n_y].flatten()))
        #     loglike = loglike - penalty
        
        if Ridge:
            penalty = lambda2_ * np.sqrt(np.sum(np.abs((T_t[:n_y,:n_y]-np.eye(n_y))).flatten()**2 ))   # penalizing by thelead-lag parameters
            loglike = loglike - penalty
            
        # elif Ridge and Full:
        #     penalty = lambda2_ * np.sqrt(np.sum(np.abs((T_t[:n_y,:n_y]-np.eye(n_y))).flatten()**2 ))   # penalizing by thelead-lag parameters
        #     penalty += lambda_ * np.sum(np.abs(Ht.flatten()**2))
        #     penalty += lambda_ * np.sum(np.abs(Qt[:n_y,:n_y].flatten()))
        #     loglike = loglike - penalty
  
    return att, Ptt, at, Pt, vt, Ft, Kt, loglike


def LeadLagSmoothing(Y, Z_t, T_t, att, Ptt, Pt, vt, Ft, Kt):

    # smoothing
    
    x_smooth = np.zeros_like(att)
    V_smooth = np.zeros_like(Ptt)
    Vt_smooth = np.zeros_like(Ptt)
    Jtn = np.zeros_like(Ptt)
    
    # x_smooth = np.empty(np.shape(att))
    x_smooth[-1] = att[-1]
    
    Jtn_0 = Ptt[-1]@ np.transpose(T_t) @np.linalg.inv(Pt[-1])
    
    N = len(Y)
    # M = np.shape(Jtn_0)[0] 
    # K = np.shape(Jtn_0)[1]
    
    # Jtn = np.empty((N-1, M, K))
    Jtn[-1] = Jtn_0
    
    x_smooth[-2]= att[-2]+Jtn[-1]@(x_smooth[-1]-T_t@att[-2])
    
    # The aoutocorrelation = Vt_smooth
    Vt_smooth_n = (np.eye(len(att[0])) - Kt[-1]@Z_t) @ T_t @ Ptt[-2]
    
    # M = np.shape(Vt_smooth_n)[0] 
    # K = np.shape(Vt_smooth_n)[1]
    
    # Vt_smooth = np.empty((N,M,K))
    Vt_smooth[-1] = Vt_smooth_n
    
    # Vt
    
    V_smooth_n = Ptt[-1]
    V_smooth_n1 = Ptt[-2] + Jtn[-1]@(V_smooth_n - Pt[-1])@Jtn[-1]
    
    # M = np.shape(V_smooth_n)[0] 
    # K = np.shape(V_smooth_n)[1]
    
    # V_smooth = np.empty((N,M,K))
    V_smooth[-1] = V_smooth_n
    V_smooth[-2] = V_smooth_n1
    
    for i in range(N-2,0,-1):
        
        # Jtn[i-1] = Ptt[i-1]@ np.transpose(T_t) @np.linalg.inv(Pt[i])
        temp = T_t @ Ptt[i-1].T
        Jtn[i-1] = np.linalg.solve(Pt[i].T, temp).T
        
        x_smooth[i-1]= att[i-1]+Jtn[i-1]@(x_smooth[i]-T_t@att[i-1])
        V_smooth[i-1] = Ptt[i-1] + Jtn[i-1]@(V_smooth[i] - Pt[i])@Jtn[i-1]
        Vt_smooth[i] = Ptt[i] @ np.transpose(Jtn[i-1]) + Jtn[i] @ (Vt_smooth[i+1] - T_t@Ptt[i])@np.transpose(Jtn[i-1]) 

    return x_smooth, V_smooth, Vt_smooth 

# Smooting function for the Lead-Lag model optimized for numba
@njit
def FastLeadLagSmoothing(Y, Z_t, T_t, att, Ptt, Pt, vt, Ft, Kt):
    
    Y   = Y.astype(np.float64)
    Z_t = Z_t.astype(np.float64)
    T_t = T_t.astype(np.float64)
    att  = att.astype(np.float64)
    Ptt  = Ptt.astype(np.float64)
    vt  = vt.astype(np.float64)
    Pt  = Pt.astype(np.float64) 
    Kt  = Kt.astype(np.float64)
    Ft  = Ft.astype(np.float64)
    
    # smoothing
    
    x_smooth = np.empty(np.shape(att))
    x_smooth[-1] = att[-1]
    
    # Jtn_0 = Ptt[-1]@ np.transpose(T_t) @np.linalg.inv(Pt[-1])
    tmp = Ptt[-1] @ T_t.T
    Jtn_0 = np.linalg.solve(Pt[-1].T, tmp.T).T

    
    N = len(Y)
    M = np.shape(Jtn_0)[0] 
    K = np.shape(Jtn_0)[1]
    
    Jtn = np.empty((N-1, M, K))
    Jtn[-1] = Jtn_0
    
    x_smooth[-2]= att[-2]+Jtn[-1]@(x_smooth[-1]-T_t@att[-2])
    
    # The aoutocorrelation = Vt_smooth
    Vt_smooth_n = (np.eye(len(att[0])) - Kt[-1]@Z_t) @ T_t @ Ptt[-2]
    
    M = np.shape(Vt_smooth_n)[0] 
    K = np.shape(Vt_smooth_n)[1]
    
    Vt_smooth = np.empty((N,M,K))
    Vt_smooth[-1] = Vt_smooth_n
    
    # Vt
    
    V_smooth_n = Ptt[-1]
    V_smooth_n1 = Ptt[-2] + Jtn[-1]@(V_smooth_n - Pt[-1])@Jtn[-1]
    
    M = np.shape(V_smooth_n)[0] 
    K = np.shape(V_smooth_n)[1]
    
    V_smooth = np.empty((N,M,K))
    V_smooth[-1] = V_smooth_n
    V_smooth[-2] = V_smooth_n1
    
    for i in range(N-2,0,-1):
        
        # Jtn[i-1] = Ptt[i-1]@ np.transpose(T_t) @np.linalg.inv(Pt[i]) 
        tmp = Ptt[i-1] @ T_t.T
        Jtn[i-1] = np.linalg.solve(Pt[i].T, tmp.T).T

        x_smooth[i-1]= att[i-1]+Jtn[i-1]@(x_smooth[i]-T_t@att[i-1])
        V_smooth[i-1] = Ptt[i-1] + Jtn[i-1]@(V_smooth[i] - Pt[i])@Jtn[i-1]
        Vt_smooth[i] = Ptt[i] @ np.transpose(Jtn[i-1]) + Jtn[i] @ (Vt_smooth[i+1] - T_t@Ptt[i])@np.transpose(Jtn[i-1]) 

    return x_smooth, V_smooth, Vt_smooth 

def LeadLagObjFun(params, yt, burnIn=10, K=None, Lasso=False, lambda_ = .5, Ridge=False, lamda2_=.5):
    
    n_var = len(yt[0])
    # n_corr = int(factorial(n_var)/(2*factorial(n_var-2)))
    
    if K==None:
        leadlag_par = params[:n_var*n_var]
    elif K==-1:
        leadlag_par = 1/np.sqrt(2*pi) * np.exp(-.5*params[:n_var*n_var]**2)
    else:
        leadlag_par = K * 2/pi * np.arctan(params[:n_var*n_var])
        
        for i in range(0, n_var*n_var, n_var+1):
            leadlag_par[i] = 1/np.sqrt(2*pi) * np.exp(-.5*params[i]**2)
        
        # leadlag_par[-1] = 1/np.sqrt(2*pi) * np.exp(.5*params[:n_var*n_var]**2)
    
    H_sigmas = np.exp(params[n_var*n_var : n_var*n_var + n_var])
    Q_sigmas = np.exp(params[n_var*n_var + n_var : n_var*n_var + n_var*2])
    Q_rhos = sigmoid_like_fun(params[n_var*n_var + n_var*2 : ])
    
    ####### The lead lag matrix
    
    F = np.reshape(leadlag_par, (n_var, n_var))
    
    Tsu = np.hstack((np.eye(n_var) + F, -F))
    Tgiu = np.hstack((np.eye(n_var), np.zeros((n_var, n_var))))
    T_t = np.vstack((Tsu, Tgiu)) # Final lead-lag matrix (Phi in the paper/thesis)
    
    # H_params = np.hstack((H_sigmas, H_rhos))
    Q_params = np.hstack((Q_sigmas, Q_rhos))
    
    H = np.diag(H_sigmas)
    Q = matrix_generator(Q_params.flatten(), n_var)
    
    Q_t = np.hstack((Q, np.zeros(np.shape(Q))))
    Q_t = np.vstack((Q_t, np.zeros((len(Q_t),len(Q_t[0]*2)))))
    
    Z_t = np.hstack((np.eye(n_var), np.zeros((n_var, n_var))))
    # print(H)
    
    att, Ptt, at, Pt, vt, Ft, Kt, loglike = FastLeadLagKF(yt, Z_t, T_t, H, Q_t, burnIn, Lasso, lambda_, Ridge, lamda2_)
    # print(loglike)
    return -loglike

def LeadLagObjFun2(params, yt, burnIn=10, K=None, Lasso=False, lambda_ = .5, Ridge=False, lamda2_=.5):
    # This functionis for fitting the restricted model without cross lead-lag effects
    n_var = len(yt[0])
    
    if K==None:
        leadlag_par = params[:n_var]
    elif K==-1:
        leadlag_par = 1/np.sqrt(2*pi) * np.exp(-.5*params[:n_var]**2)
    else:
        leadlag_par = K * 2/pi * np.arctan(params[:n_var] )
    
    H_sigmas = np.exp(params[n_var : n_var + n_var])
    Q_sigmas = np.exp(params[n_var + n_var : n_var*n_var + n_var*2])
    Q_rhos = sigmoid_like_fun(params[n_var + n_var*2 : ])
    
    ####### The lead lag matrix
    
    F = np.diag(leadlag_par)
    
    Tsu = np.hstack((np.eye(n_var) + F, -F))
    Tgiu = np.hstack((np.eye(n_var), np.zeros((n_var, n_var))))
    T_t = np.vstack((Tsu, Tgiu)) # Final lead-lag matrix (Phi in the paper/thesis)
    
    # H_params = np.hstack((H_sigmas, H_rhos))
    Q_params = np.hstack((Q_sigmas, Q_rhos))
    
    H = np.diag(H_sigmas)
    Q = matrix_generator(Q_params.flatten(), n_var)
    
    Q_t = np.hstack((Q, np.zeros(np.shape(Q))))
    Q_t = np.vstack((Q_t, np.zeros((len(Q_t),len(Q_t[0]*2)))))
    
    Z_t = np.hstack((np.eye(n_var), np.zeros((n_var, n_var))))
    # print(H)
    
    # att, Ptt, at, Pt, vt, Ft, Kt, loglike = LeadLagKF(yt, Z_t, T_t, H, Q_t, burnIn=10)
    att, Ptt, at, Pt, vt, Ft, Kt, loglike = FastLeadLagKF(yt, Z_t, T_t, H, Q_t, burnIn, Lasso, lambda_, Ridge, lamda2_)
    
    return -loglike

def LeadLagMLfit(params, yt, n_var, model=1, burnIn=10, K=None, Lasso=False, lambda_ = .5, Ridge=False, lambda2_=.5):
    
    if model==1:
        res = minimize(LeadLagObjFun, params, args=(yt, burnIn, K, Lasso, lambda_, Ridge, lambda2_), method='BFGS')
        
        fitted_params = res.x 
        
        if K==None:
            leadlag_par = fitted_params[:n_var*n_var]
        elif K==-1:
            leadlag_par = 1/np.sqrt(2*pi) * np.exp(-.5*fitted_params[:n_var*n_var]**2)
        else:
            leadlag_par = K * 2/pi * np.arctan(fitted_params[:n_var*n_var])
                        
            for i in range(0, n_var*n_var, n_var+1):
                leadlag_par[i] = 1/np.sqrt(2*pi) * np.exp(-.5*fitted_params[i]**2)
            
        H_sigmas = np.exp(fitted_params[n_var*n_var : n_var*n_var + n_var])
        Q_sigmas = np.exp(fitted_params[n_var*n_var + n_var : n_var*n_var + n_var*2])
        Q_rhos = sigmoid_like_fun(fitted_params[n_var*n_var + n_var*2 : ])
        
        ####### The lead lag matrix
        F = np.reshape(leadlag_par, (n_var, n_var))
        
    elif model==2:
        
        res = minimize(LeadLagObjFun2, params, args=(yt, burnIn, K, Lasso, lambda_, Ridge, lambda2_), method='BFGS')
        
        fitted_params = res.x 
        
        # leadlag_par = fitted_params[:n_var]
        
        if K==None:
            leadlag_par = fitted_params[:n_var]
        elif K==-1:
            leadlag_par = 1/np.sqrt(2*pi) * np.exp(.5*fitted_params[:n_var]**2)
        else:
            leadlag_par = K * 2/pi * np.arctan(fitted_params[:n_var])
        
        H_sigmas = np.exp(fitted_params[n_var : n_var + n_var])
        Q_sigmas = np.exp(fitted_params[n_var + n_var : n_var*n_var + n_var*2])
        Q_rhos = sigmoid_like_fun(fitted_params[n_var + n_var*2 : ])
        
        ####### The lead lag matrix
        F = np.diag(leadlag_par)
        # print(np.shape(F))
    
    else:
        raise ValueError('model==1 or model==2')
    
    Tsu = np.hstack((np.eye(n_var) + F, -F))
    Tgiu = np.hstack((np.eye(n_var), np.zeros((n_var, n_var))))
    T_t = np.vstack((Tsu, Tgiu)) # Final lead-lag matrix (Phi in the paper/thesis)
    
    # H_params = np.hstack((H_sigmas, H_rhos))
    Q_params = np.hstack((Q_sigmas, Q_rhos))
    
    H = np.diag(H_sigmas)
    Q = matrix_generator(Q_params.flatten(), n_var)
    # print('ciao')
    return F, H, Q, T_t

def LeadLagllem(y, Q_init, R_init, F_init, C, maxiter=3000, eps=10**-4):
    
    # Matrix d*T d=number of variables, T=number of observations 
    
    # Z_t = C
    
    # Q_init = Q_t
    # R_init = R_t
    # F_init = T_t
    # C = Z_t
    
    Q = Q_init
    R = R_init
    F = F_init
    
    [T, d] = np.shape(y)
    
    Id = np.eye(d)
    
    P = np.hstack((Id, np.zeros((d,d))))
    
    burnIn = d + 10
    
    l = np.ones(maxiter+1)*(10**10)
        
    l_old = 10**10
    
    ###
    for i in range(1, maxiter+1):
    
        S = np.zeros((2*d, 2*d))
        S10 = np.zeros((2*d, 2*d))
        eps_smooth = np.zeros((d, d))
                
        att, Ptt, at, Pt, vt, Ft, Kt, loglike = LeadLagKF(y, C, F, R, Q, burnIn)
        # at, att, Pt, Ptt, vt, Ft, Kt, loglike = LeadLagKF2(y, Q, R, F, C, 1e-8, False)
        x_smooth, V_smooth, Vt_smooth = LeadLagSmoothing(y, C, F, att, Ptt, Pt, vt, Ft, Kt)
        
        for t in range(burnIn, T):
            
            l[i] = loglike
            l_old = abs(l[i-1] - loglike)
            
            if True in np.isnan(y[t,:]):
                
                # nan_index = np.where(np.isnan(y[t,:]))
                no_nan_index = np.where(~np.isnan(y[t,:]))
                
                y1 = np.zeros(np.shape(copy(y[t,:])))
                y1[no_nan_index] = copy(y[t,:][no_nan_index])
                y1 = np.reshape(y1, (len(y1),1))
                
                x = x_smooth[t,:]
                x = np.reshape(x, (len(x),1))
                
                x1 = x_smooth[t-1,:]
                x1 = np.reshape(x1, (len(x1),1))
                
                S = S + x @ x.T + V_smooth[t]
                S10 = S10 + x1 @ x1.T + Vt_smooth[t]
                
                auxY = np.copy(y1)
                auxY = copy(y1)
                
                auxC = np.zeros(np.shape(C))
                auxC[no_nan_index] = copy(C[no_nan_index])
                
                R2 = copy(R)
                R2[no_nan_index] = 0
                
                eps_smooth += (auxY - auxC@x) @ (auxY - auxC@x).T + auxC@V_smooth[t]@auxC.T + R2
                
            else:
                
                x = x_smooth[t,:]
                x = np.reshape(x, (len(x),1))
                
                x1 = x_smooth[t-1,:]
                x1 = np.reshape(x1, (len(x1),1))
                
                S = S + x @ x.T + V_smooth[t]
                S10 = S10 + x1 @ x1.T + Vt_smooth[t]
                
                auxY = np.reshape(y[t,:], (len(y[t,:]),1))
                
                eps_smooth += (auxY - C@x) @ (auxY - C@x).T + C@V_smooth[t]@C.T
        
        x = x_smooth[t-1,:]
        x = np.reshape(x, (len(x),1))
        
        x1 = x_smooth[burnIn-1,:]
        x1 = np.reshape(x1, (len(x1),1))
        
        S00 = S - x@x.T - V_smooth[T-1] + x1 @ x1.T + V_smooth[burnIn-1]
        S11 = deepcopy(S)
        
        BB = deepcopy(S10)
        AA = deepcopy(S00)
        CC = deepcopy(S11)
        
        BB1 = BB[0:d, 0:d]
        BB2 = BB[0:d, d:]
        BBtilde = np.hstack((BB1, BB2))
        
        AA11 = AA[0:d, 0:d]
        AA12 = AA[0:d, d:]
        AA21 = AA[d:, 0:d]
        AA22 = AA[d:, d:]
        
        Gamma = BB1 - BB2 - AA11 + AA12
        Theta = AA11 + AA22 - AA12 - AA21
        
        auxF = Gamma @ np.linalg.inv(Theta)
        Ftilde = np.hstack((Id + auxF, -auxF))
        
        auxQ = P@CC@P.T - BBtilde@Ftilde.T - Ftilde@BBtilde.T + Ftilde@AA@Ftilde.T
        auxQ = auxQ/(T-burnIn+1)
        
        temp_Q = np.tril(auxQ) + np.tril(auxQ, -1).T
        Q = np.hstack((temp_Q, np.zeros((d,d))))
        Q = np.vstack((Q, np.zeros((d, 2*d))))
        
        Fgiu = np.hstack((Id, np.zeros((d,d))))
        
        F = np.vstack((Ftilde, Fgiu))
        
        temp_diag_R = np.diag(eps_smooth/(T-burnIn+1)).copy()
        # print(temp_diag_R)
        # print(np.where(np.isnan(temp_diag_R)))
        
        if True in np.isnan(temp_diag_R):
            temp_diag_R[np.where(np.isnan(temp_diag_R))] = 1
        elif 0 in temp_diag_R:
            temp_diag_R[temp_diag_R==0] = 1
        
        R = np.diag(temp_diag_R)
        
        
        if l_old<=eps:
            print('Tolerance level: '+str(l_old))
            print('N. of iteration: '+str(i))
            break
            
        if i==maxiter:
            print('Tolerance level: '+str(l_old))
            print('N. of iteration: '+str(i))
    
    return F, R, Q, att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, vt, Ft, loglike


def Fast_LeadLagllem(y, Q_init, R_init, F_init, C, maxiter=3000, eps=1e-4):
    # === keep math identical, only add defensive checks & tiny regularizers ===

    # copies and ensure float64
    Q = Q_init.copy().astype(np.float64)
    R = R_init.copy().astype(np.float64)   # <-- important: ensure R is float64
    F = F_init.copy().astype(np.float64)

    y = y.astype(np.float64)

    T, d = y.shape
    Id = np.eye(d)
    P = np.hstack((Id, np.zeros((d, d))))
    burnIn = d + 10

    l = np.ones(maxiter + 1) * 1e10
    l_old = 1e10

    # small jitter to keep matrices PD (non-invasive)
    jitter = 1e-12

    # helper: sanitize a square matrix in-place
    def _sanitize_square(mat):
        # make finite
        if np.any(~np.isfinite(mat)):
            # replace non-finite with small diagonal
            mat[:] = 0.0
            for ii in range(mat.shape[0]):
                mat[ii, ii] = jitter
        # symmetrize small asymmetries
        mat[:] = 0.5 * (mat + mat.T)

    # initial sanitize
    _sanitize_square(Q[:d, :d])
    _sanitize_square(R)

    for i in range(1, maxiter + 1):
        S = np.zeros((2 * d, 2 * d))
        S10 = np.zeros((2 * d, 2 * d))
        eps_smooth = np.zeros((d, d))

        # Call the KF / smoother (preserve same API)
        att, Ptt, at, Pt, vt, Ft, Kt, loglike = FastLeadLagKF(y, C, F, R, Q, burnIn)
        # defensive check: ensure returned matrices are finite
        # check a few important arrays: Pt, Ptt, Ft
        if np.any(~np.isfinite(Pt)):
            raise ValueError("NaN/Inf detected in Pt returned by FastLeadLagKF. "
                             "Inspect Pt (shapes: Pt.shape=%s)."
                             % (str(Pt.shape),))
        if np.any(~np.isfinite(Ptt)):
            raise ValueError("NaN/Inf detected in Ptt returned by FastLeadLagKF. "
                             "Inspect Ptt (shapes: Ptt.shape).")
        if np.any(~np.isfinite(Ft)):
            raise ValueError("NaN/Inf detected in Ft returned by FastLeadLagKF. "
                             "Inspect Ft (shapes: Ft.shape).")

        x_smooth, V_smooth, Vt_smooth = FastLeadLagSmoothing(y, C, F, att, Ptt, Pt, vt, Ft, Kt)

        for t in range(burnIn, T):
            l[i] = loglike
            l_old = abs(l[i - 1] - loglike)

            # detect NaNs in observation row
            has_nan = False
            for jj in range(d):
                if np.isnan(y[t, jj]):
                    has_nan = True
                    break

            if has_nan:
                # build y1 with available entries
                y1 = np.zeros((d, 1))
                for jj in range(d):
                    if not np.isnan(y[t, jj]):
                        y1[jj, 0] = y[t, jj]

                x = x_smooth[t, :].reshape((2 * d, 1))
                x1 = x_smooth[t - 1, :].reshape((2 * d, 1))

                S += x @ x.T + V_smooth[t]
                S10 += x1 @ x1.T + Vt_smooth[t]

                auxC = np.zeros(C.shape)
                for jj in range(d):
                    if not np.isnan(y[t, jj]):
                        auxC[jj, :] = C[jj, :]

                R2 = R.copy()
                for jj in range(d):
                    if not np.isnan(y[t, jj]):
                        R2[jj, jj] = 0.0

                auxY = y1
                eps_smooth += (auxY - auxC @ x) @ (auxY - auxC @ x).T + auxC @ V_smooth[t] @ auxC.T + R2

            else:
                x = x_smooth[t, :].reshape((2 * d, 1))
                x1 = x_smooth[t - 1, :].reshape((2 * d, 1))
                S += x @ x.T + V_smooth[t]
                S10 += x1 @ x1.T + Vt_smooth[t]
                auxY = y[t, :].reshape((d, 1))
                eps_smooth += (auxY - C @ x) @ (auxY - C @ x).T + C @ V_smooth[t] @ C.T

        # NOTE: keep identical indexing/shape semantics as original
        x = x_smooth[T - 2, :].reshape((2 * d, 1))
        x1 = x_smooth[burnIn - 1, :].reshape((2 * d, 1))

        S00 = S - x @ x.T - V_smooth[T - 1] + x1 @ x1.T + V_smooth[burnIn - 1]
        S11 = S.copy()
        BB = S10.copy()
        AA = S00.copy()
        CC = S11.copy()

        BB1 = BB[0:d, 0:d]
        BB2 = BB[0:d, d:]
        BBtilde = np.hstack((BB1, BB2))

        AA11 = AA[0:d, 0:d]
        AA12 = AA[0:d, d:]
        AA21 = AA[d:, 0:d]
        AA22 = AA[d:, d:]

        Gamma = BB1 - BB2 - AA11 + AA12
        Theta = AA11 + AA22 - AA12 - AA21

        # small safeguard: if Theta has NaNs/Infs, replace them with tiny diag
        if np.any(~np.isfinite(Theta)):
            Theta[:] = 0.0
            for jj in range(d):
                Theta[jj, jj] = jitter

        # invert Theta (if near-singular, add tiny jitter)
        try:
            invTheta = np.linalg.inv(Theta)
        except np.linalg.LinAlgError:
            Theta = Theta + jitter * np.eye(d)
            invTheta = np.linalg.inv(Theta)

        auxF = Gamma @ invTheta
        Ftilde = np.hstack((Id + auxF, -auxF))

        auxQ = P @ CC @ P.T - BBtilde @ Ftilde.T - Ftilde @ BBtilde.T + Ftilde @ AA @ Ftilde.T
        denom = float((T - burnIn + 1))
        if denom == 0.0:
            denom = 1.0
        auxQ = auxQ / denom

        # symmetrize auxQ and enforce finite
        if np.any(~np.isfinite(auxQ)):
            # fallback: small diagonal
            auxQ = np.zeros((2 * d, 2 * d))
            for jj in range(d):
                auxQ[jj, jj] = jitter
        auxQ = 0.5 * (auxQ + auxQ.T)

        # keep same shapes as original
        temp_Q = np.tril(auxQ[:d, :d]) + np.tril(auxQ[:d, :d], -1).T
        Q = np.hstack((temp_Q, np.zeros((d, d))))
        Q = np.vstack((Q, np.zeros((d, 2 * d))))

        Fgiu = np.hstack((Id, np.zeros((d, d))))
        F = np.vstack((Ftilde, Fgiu))

        # update R diag
        denom2 = float((T - burnIn + 1))
        if denom2 == 0.0:
            denom2 = 1.0
        # temp_diag_R = np.diag(eps_smooth / denom2)

        # temp_diag_R might be a full matrix if eps_smooth/... gives matrix; extract diagonal safely
        # temp_diag_vec = np.zeros(d)
        # for jj in range(d):
        #     val = temp_diag_R[jj, jj] if np.isfinite(temp_diag_R[jj, jj]) else 0.0
        #     if val == 0.0:
        #         val = 1.0
        #     temp_diag_vec[jj] = val
        
        # Compute diagonal vector safely
        temp_diag_vec = np.zeros(d)
        eps_diag = np.diag(eps_smooth / denom2)  # this is 1D now
        for jj in range(d):
            val = eps_diag[jj] if np.isfinite(eps_diag[jj]) and eps_diag[jj] != 0.0 else 1.0
            temp_diag_vec[jj] = val
        
        # Build final diagonal matrix
        R = np.diag(temp_diag_vec)
        
        # final sanitize Q/R
        _sanitize_square(Q[:d, :d])
        _sanitize_square(R)

        if l_old <= eps:
            break

    return F, R, Q, att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, vt, Ft, loglike


@njit
def Fast_LeadLagllem_numba(y, Q_init, R_init, F_init, C, maxiter=3000, eps=1e-4):
    # Numba-compatible EM for Lead-Lag state-space model.
    # Dimensions and math identical to original LeadLagllem.
    # Adds defensive checks and jitter to avoid NaNs/Infs.

    # Ensure float64 copies
    Q = Q_init.astype(np.float64).copy()
    R = R_init.astype(np.float64).copy()
    F = F_init.astype(np.float64).copy()
    y = y.astype(np.float64).copy()

    T, d = y.shape
    Id = np.eye(d)
    P = np.hstack((Id, np.zeros((d, d))))
    burnIn = d + 10
    jitter = 1e-12

    l_old = 1e10

    # Helper: sanitize a matrix in-place
    def sanitize_square(mat):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isfinite(mat[i, j]):
                    mat[i, j] = 0.0
            mat[i, i] += jitter  # add tiny jitter on diagonal

    # Initial sanitize
    sanitize_square(Q[:d, :d])
    sanitize_square(R)

    # EM loop
    for it in range(maxiter):
        S = np.zeros((2*d, 2*d))
        S10 = np.zeros((2*d, 2*d))
        eps_smooth = np.zeros((d, d))

        # === Call KF / Smoother ===
        att, Ptt, at, Pt, vt, Ft, Kt, loglike = FastLeadLagKF(y, C, F, R, Q, burnIn)
        x_smooth, V_smooth, Vt_smooth = FastLeadLagSmoothing(y, C, F, att, Ptt, Pt, vt, Ft, Kt)

        # Sanitize smoother outputs
        for t in range(T):
            for i in range(2*d):
                if not np.isfinite(x_smooth[t, i]):
                    x_smooth[t, i] = 0.0
            for i in range(2*d):
                for j in range(2*d):
                    if not np.isfinite(V_smooth[t][i, j]):
                        V_smooth[t][i, j] = 0.0

        # === Accumulate S, S10, eps_smooth ===
        for t in range(burnIn, T):
            # Handle missing data
            has_nan = False
            for j in range(d):
                if np.isnan(y[t, j]):
                    has_nan = True
                    break

            if has_nan:
                # Build y1 with available entries
                y1 = np.zeros((d, 1))
                auxC = np.zeros((d, 2*d))
                R2 = np.zeros((d, d))
                for j in range(d):
                    if not np.isnan(y[t, j]):
                        y1[j, 0] = y[t, j]
                        for k in range(2*d):
                            auxC[j, k] = C[j, k]
                        R2[j, j] = 0.0

                x = x_smooth[t, :].reshape((2*d, 1))
                x1 = x_smooth[t-1, :].reshape((2*d, 1))

                S += x @ x.T + V_smooth[t]
                S10 += x1 @ x1.T + Vt_smooth[t]
                eps_smooth += (y1 - auxC @ x) @ (y1 - auxC @ x).T + auxC @ V_smooth[t] @ auxC.T + R2
            else:
                x = x_smooth[t, :].reshape((2*d, 1))
                x1 = x_smooth[t-1, :].reshape((2*d, 1))
                S += x @ x.T + V_smooth[t]
                S10 += x1 @ x1.T + Vt_smooth[t]
                y_t = y[t, :].reshape((d, 1))
                eps_smooth += (y_t - C @ x) @ (y_t - C @ x).T + C @ V_smooth[t] @ C.T

        # === Update F, Q, R ===
        x = x_smooth[T-2, :].reshape((2*d, 1))
        x1 = x_smooth[burnIn-1, :].reshape((2*d, 1))

        S00 = S - x @ x.T - V_smooth[T-1] + x1 @ x1.T + V_smooth[burnIn-1]
        S11 = S.copy()
        BB = S10.copy()
        AA = S00.copy()
        CC = S11.copy()

        BB1 = BB[0:d, 0:d]
        BB2 = BB[0:d, d:]
        BBtilde = np.hstack((BB1, BB2))

        AA11 = AA[0:d, 0:d]
        AA12 = AA[0:d, d:]
        AA21 = AA[d:, 0:d]
        AA22 = AA[d:, d:]

        Gamma = BB1 - BB2 - AA11 + AA12
        Theta = AA11 + AA22 - AA12 - AA21

        sanitize_square(Theta)
        invTheta = np.linalg.inv(Theta)

        auxF = Gamma @ invTheta
        Ftilde = np.hstack((Id + auxF, -auxF))

        auxQ = P @ CC @ P.T - BBtilde @ Ftilde.T - Ftilde @ BBtilde.T + Ftilde @ AA @ Ftilde.T
        auxQ /= float(T - burnIn + 1.0)
        auxQ = 0.5*(auxQ + auxQ.T)  # symmetrize
        Q = np.vstack((np.hstack((auxQ[:d, :d], np.zeros((d,d)))), np.zeros((d, 2*d))))

        Fgiu = np.hstack((Id, np.zeros((d, d))))
        F = np.vstack((Ftilde, Fgiu))

        # Update R
        temp_diag = np.zeros(d)
        eps_smooth_diag = np.diag(eps_smooth / float(T - burnIn + 1.0))
        for j in range(d):
            val = eps_smooth_diag[j]
            if not np.isfinite(val) or val == 0.0:
                val = 1.0
            temp_diag[j] = val
        R = np.diag(temp_diag)

        sanitize_square(Q[:d, :d])
        sanitize_square(R)

        if l_old <= eps:
            break
        l_old = abs(l_old - loglike)

    return F, R, Q, att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, vt, Ft, loglike


######### VAR(1) model

def VAR1_kalman_filter(yt, A, Z, H, Q, burnIn = 0):
    
    n_var = len(yt[0]) # number of variables
    N = len(yt)
    
    Pt = np.empty((N, n_var, n_var))
    Pt[0] = np.eye(len(Q))
    
    F0 = Z @ Pt[0] @ Z.T + H # Initial valuea of the first innovartion variance
    n_rows, n_cols = np.shape(F0)
    
    Ft = np.empty((N, n_rows, n_cols))
    Ft[0] = F0
    
    
    # Ptt = [] # filtered estimator of the variance of the unobserved process
    # Ptt0 = Pt[0] - Pt[0] @ Z.T @ inv(Ft[0]) @ Z @ Pt[0]
    ZP = Z @ Pt[0]
    Ptt0 = Pt[0] - Pt[0] @ Z.T @ np.linalg.solve(Ft[0], ZP)
    n_rows, n_cols = np.shape(Ptt0)
    
    Ptt = np.empty((N, n_rows, n_cols))
    Ptt[0] = Ptt0
    
    # Kalman gain part
    # K0 = A @ Pt[0] @ Z.T @ inv(Ft[0]) # initial value of the Kalman gain
    tmp = A @ Pt[0] @ Z.T
    K0 = np.linalg.solve(Ft[0].T, tmp.T).T
    n_rows, n_cols = np.shape(K0)
    
    Kt = np.empty((N, n_rows, n_cols))
    Kt[0] = K0
    
    # Allocating the memory for at, att, vt
    
    a0 = copy(yt[0,:]) # initial value for the conditional mean of the unobserved process
    a0 = np.reshape(a0, (1, n_var))
    at = np.empty((N, n_var)) # Conditional mean of the unobserved process
    at[0] = a0
    
    vt = np.empty((N, n_var)) # innovation error
    vt0 = yt[0] - a0 # initial value of the innovation error
    
    nan_index = np.isnan(vt0)
    
    if True in nan_index:
        
        vt0[nan_index]=0
    
    vt[0] = vt0
    
    att = np.empty((N, n_var)) # filtered estimator of the unobserved process
    # att[0,:] = at[0] + Pt[0] @ Z @ inv(Ft[0]) @ vt[0] # computing the initial value of the filtered estimator of the unobserved process
    tmp = Pt[0] @ Z.T
    att[0, :] = at[0] + tmp @ np.linalg.solve(Ft[0], vt[0])
    
    for i in np.arange(1, N):
        
        nan_index = np.isnan(yt[i])
        
        if True in nan_index:
            
            y = deepcopy(yt[i])
            y[nan_index] = 0
            
            Z = deepcopy(Z)
            
            for j in range(0, len(Z)):
                
                Z[nan_index] = 0
                
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3):
                    Pt[i] = copy(Pt[i-1])
                else:
                    Pt[i] = (A @ Ptt[i-1] @ A.T + Q) # Computing the variance
            except:
                pass
            
            # Pt[i] = (A @ Ptt[i-1] @ A.T + Q) # Computing the variance
            
            auxH = np.eye(len(H))
            auxH[~nan_index] = H[~nan_index]
            
            Ft[i] = (Z @ Pt[i] @ Z.T + auxH) # computing the variance of the innovation error
            # Ft[i] = (Z@Pt[i]@np.transpose(Z) + H) # computing the variance of the innovation error
            
            F = inv(Ft[i])
                            
            Ptt[i] = (Pt[i] - Pt[i] @ Z.T @ F @ Z @Pt[i]) # computing the filtered variance
            
            at[i] = A @ att[i-1]
            vt[i] = y - Z @ at[i]
            
            att[i] = (at[i] + Pt[i] @ Z.T @ F @ np.transpose(vt[i]))
            
            Kt[i] = A @ Pt[i] @ Z.T @ F # Kalman gain
            
        else:
            
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3) and i>2:
                    Pt[i] = copy(Pt[i-1])
                    # Ft[i] = A @ Pt[i] @ A.T + H
                else:
                    Pt[i] = (A @ Ptt[i-1] @ A.T + Q) # Computing the variance
                    # Ft[i] = (Z @ Pt[i] @ Z.T + H) # computing the variance of the innovation error
            except:
                pass
        
            # Pt[i] = (A @ Ptt[i-1] @ A.T + Q) # Computing the variance
            
            Ft[i] = (Z @ Pt[i] @ Z.T + H) # computing the variance of the innovation error
          
            # F = inv(Ft[i])
            
            # Ptt[i] = (Pt[i] - Pt[i] @ Z.T @ F @ Z @ Pt[i]) # computing the filtered variance
            
            tmp = Pt[i] @ Z.T
            sol = np.linalg.solve(Ft[i], Z @ Pt[i])
            Ptt[i] = Pt[i] - tmp @ sol
            
            at[i] = A @ att[i-1]
            vt[i] = yt[i] - Z @ at[i]
            
            # att[i] = (at[i] + Pt[i] @ Z.T @ np.linalg.inv(Ft[i]) @ np.transpose(vt[i])) 
            # Solve Ft[i] * x = vt[i].T
            tmp = np.linalg.solve(Ft[i], vt[i].T)
            
            # Then multiply
            att[i] = at[i] + Pt[i] @ Z.T @ tmp
            
            # Kt[i] = A @ Pt[i] @ Z.T @ np.linalg.inv(Ft[i]) # Kalman gain 
            tmp = Z @ Pt[i].T @ A.T
            Kt[i] = np.linalg.solve(Ft[i], tmp.T).T
    
    loglike = 0
    n_var = len(yt[0])
    
    for i in range(burnIn, len(vt)):
        
        loglike += multivariate_gaussian_log_likelihood(vt[i], np.zeros(n_var), Ft[i])
    
    return att, Ptt, at, Pt, vt, Ft, Kt, loglike  

@njit
def FastVAR1_kalman_filter(yt, A, Z, H, Q, burnIn=0, Lasso=False, lambda_ = .5, Ridge=False, lamda2_=.5):
    
    yt = yt.astype(np.float64)
    A = A.astype(np.float64) 
    Z = Z.astype(np.float64)
    H = H.astype(np.float64)
    Q = Q.astype(np.float64)
    
    n_var = yt.shape[1]
    N = yt.shape[0]
    
    Pt = np.zeros((N, n_var, n_var))
    Pt[0] = np.eye(len(Q))
    
    F0 = Z @ Pt[0] @ np.transpose(Z) + H
    Ft = np.zeros((N, F0.shape[0], F0.shape[1]))
    Ft[0] = F0
    
    # Initial filtered covariance
    ZP = Z @ Pt[0]
    Ptt0 = Pt[0] - Pt[0] @ np.transpose(Z) @ np.linalg.solve(Ft[0], ZP)
    Ptt = np.zeros((N, Ptt0.shape[0], Ptt0.shape[1]))
    Ptt[0] = Ptt0
    
    # Initial Kalman gain
    tmp = A @ Pt[0] @ np.transpose(Z)
    K0 = np.linalg.solve(Ft[0].T, tmp.T).T
    Kt = np.zeros((N, K0.shape[0], K0.shape[1]))
    Kt[0] = K0
    
    at = np.zeros((N, n_var))
    att = np.zeros((N, n_var))
    vt = np.zeros((N, n_var))
    
    a0 = yt[0]
    at[0] = a0
    
    vt0 = yt[0] - a0
    for j in range(n_var):
        if np.isnan(vt0[j]):
            vt0[j] = 0.0
    vt[0] = vt0
    
    tmp0 = Pt[0] @ np.transpose(Z)
    att[0] = at[0] + tmp0 @ np.linalg.solve(Ft[0], vt[0])
    
    for i in range(1, N):
        nan_index = np.isnan(yt[i])
    
        if np.any(nan_index):
            y = np.copy(yt[i])
            for j in range(n_var):
                if nan_index[j]:
                    y[j] = 0.0
    
            Z_mod = np.copy(Z)
            for j in range(Z.shape[0]):
                for k in range(Z.shape[1]):
                    if nan_index[j]:
                        Z_mod[j, k] = 0.0
    
            if i >= 2:
                m = np.mean(Pt[i-1] - Pt[i-2])
                if abs(m) < 1e-3:
                    Pt[i] = Pt[i-1]
                else:
                    Pt[i] = A @ Ptt[i-1] @ A.T + Q
            else:
                Pt[i] = A @ Ptt[i-1] @ A.T + Q
    
            auxH = np.eye(H.shape[0])
            for j in range(H.shape[0]):
                if not nan_index[j]:
                    auxH[j, j] = H[j, j]
    
            Ft[i] = Z_mod @ Pt[i] @ np.transpose(Z_mod) + auxH
    
            sol = np.linalg.solve(Ft[i], Z_mod @ Pt[i])
            Ptt[i] = Pt[i] - Pt[i] @ np.transpose(Z_mod) @ sol
    
            at[i] = A @ att[i-1]
            vt[i] = y - Z_mod @ at[i]
            att[i] = at[i] + Pt[i] @ np.transpose(Z_mod) @ np.linalg.solve(Ft[i], vt[i])
    
            tmp = Z_mod @ Pt[i].T @ A.T
            Kt[i] = np.linalg.solve(Ft[i], tmp.T).T
    
        else:
            if i >= 2:
                m = np.mean(Pt[i-1] - Pt[i-2])
                if abs(m) < 1e-3:
                    Pt[i] = Pt[i-1]
                else:
                    Pt[i] = A @ Ptt[i-1] @ A.T + Q
            else:
                Pt[i] = A @ Ptt[i-1] @ A.T + Q
    
            Ft[i] = Z @ Pt[i] @ np.transpose(Z) + H
    
            sol = np.linalg.solve(Ft[i], Z @ Pt[i])
            Ptt[i] = Pt[i] - Pt[i] @ np.transpose(Z) @ sol
    
            at[i] = A @ att[i-1]
            vt[i] = yt[i] - Z @ at[i]
            att[i] = at[i] + Pt[i] @ np.transpose(Z) @ np.linalg.solve(Ft[i], vt[i])
    
            tmp = Z @ Pt[i].T @ A.T
            Kt[i] = np.linalg.solve(Ft[i], tmp.T).T
    
    loglike = 0.0
    for i in range(burnIn, N):
        loglike += Fast_multivariate_gaussian_log_likelihood(vt[i], np.zeros(n_var), Ft[i])
    
    return att, Ptt, at, Pt, vt, Ft, Kt, loglike

def VAR1Smoothing(Y, A, Z, att, Ptt, Pt, vt, Ft, Kt):

    N = len(Y)
    
    x_smooth = np.empty(np.shape(att))
    x_smooth[-1] = att[-1]
    
    # Jtn_0 = Ptt[-1]@ np.transpose(A) @np.linalg.inv(Pt[-1])
    Jtn_0 = np.linalg.solve(Pt[-1].T, (Ptt[-1] @ A).T).T
    
    n_rows, n_cols = np.shape(Jtn_0) 
    
    Jtn = np.empty((N-1, n_rows, n_cols))
    Jtn[-1] = Jtn_0
    
    x_smooth[-2]= att[-2] + Jtn[-1] @ (x_smooth[-1] - A @ att[-2])
    
    # The aoutocorrelation = Vt_smooth
    Vt_smooth_n = (np.eye(len(att[0])) - Kt[-1] @ Z) @ A @ Ptt[-2]
    
    n_rows, n_cols = np.shape(Vt_smooth_n) 
    
    Vt_smooth = np.empty((N, n_rows, n_cols))
    Vt_smooth[-1] = Vt_smooth_n
    
    # Vt
    V_smooth_n = Ptt[-1]
    V_smooth_n1 = Ptt[-2] + Jtn[-1] @ (V_smooth_n - Pt[-1]) @ Jtn[-1]
    
    n_rows, n_cols = np.shape(V_smooth_n) 
    
    V_smooth = np.empty((N, n_rows, n_cols))
    V_smooth[-1] = V_smooth_n
    V_smooth[-2] = V_smooth_n1 

    for i in range(N-2,0,-1):
        
        # Jtn[i-1] = Ptt[i-1]@ np.transpose(A) @np.linalg.inv(Pt[i])
        Jtn[i-1] = np.linalg.solve(Pt[i].T, (Ptt[i-1] @ A).T).T
        x_smooth[i-1]= att[i-1]+Jtn[i-1]@(x_smooth[i]-A@att[i-1])
        V_smooth[i-1] = Ptt[i-1] + Jtn[i-1]@(V_smooth[i] - Pt[i])@Jtn[i-1]
        Vt_smooth[i] = Ptt[i] @ np.transpose(Jtn[i-1]) + Jtn[i] @ (Vt_smooth[i+1] - A@Ptt[i])@np.transpose(Jtn[i-1]) 

    return x_smooth, V_smooth, Vt_smooth 

@njit
def FastVAR1Smoothing(Y, A, Z, att, Ptt, Pt, vt, Ft, Kt):
    N = len(Y)
    
    # smoothed states
    x_smooth = np.empty_like(att)
    x_smooth[-1] = att[-1]
    
    # initial Jtn
    Jtn_0 = np.linalg.solve(Pt[-1].T, (Ptt[-1] @ A).T).T
    n_rows, n_cols = Jtn_0.shape
    Jtn = np.empty((N-1, n_rows, n_cols))
    Jtn[-1] = Jtn_0
    
    # second to last smoother update
    x_smooth[-2] = att[-2] + Jtn[-1] @ (x_smooth[-1] - A @ att[-2])
    
    # Vt_smooth
    Vt_smooth_n = (np.eye(att.shape[1]) - Kt[-1] @ Z) @ A @ Ptt[-2]
    n_rows, n_cols = Vt_smooth_n.shape
    Vt_smooth = np.empty((N, n_rows, n_cols))
    Vt_smooth[-1] = Vt_smooth_n
    
    # V_smooth
    V_smooth_n = Ptt[-1]
    V_smooth_n1 = Ptt[-2] + Jtn[-1] @ (V_smooth_n - Pt[-1]) @ Jtn[-1]
    
    n_rows, n_cols = V_smooth_n.shape
    V_smooth = np.empty((N, n_rows, n_cols))
    V_smooth[-1] = V_smooth_n
    V_smooth[-2] = V_smooth_n1
    
    # backward recursion
    for i in range(N-2, 0, -1):
        Jtn[i-1] = np.linalg.solve(Pt[i].T, (Ptt[i-1] @ A).T).T
        x_smooth[i-1] = att[i-1] + Jtn[i-1] @ (x_smooth[i] - A @ att[i-1])
        V_smooth[i-1] = Ptt[i-1] + Jtn[i-1] @ (V_smooth[i] - Pt[i]) @ Jtn[i-1]
    
        # cross-covariance update
        Vt_smooth[i] = Ptt[i] @ Jtn[i-1].T + Jtn[i-1] @ (Vt_smooth[i+1] - A @ Ptt[i]) @ Jtn[i-1].T
        
    return x_smooth, V_smooth, Vt_smooth

def VAR1_obj_fun(params, yt, Lasso=False, lambda_ = .5, Ridge=False, lamda2_=.5):
    
    n_var = len(yt[0])
    n_corr = int(factorial(n_var)/(2*factorial(n_var-2)))
    
    autoreg_par = params[:n_var*n_var]
    H_sigmas = np.exp(params[n_var*n_var : n_var*n_var + n_var])
    Q_sigmas = np.exp(params[n_var*n_var + n_var : n_var*n_var + n_var*2])
    H_rhos = sigmoid_like_fun(params[n_var*n_var + n_var*2 : n_var*n_var + n_var*2 + n_corr])
    Q_rhos = sigmoid_like_fun(params[n_var*n_var + n_var*2 + n_corr :])
    
    A = np.reshape(autoreg_par, (n_var, n_var))
    
    H_params = np.hstack((H_sigmas, H_rhos))
    Q_params = np.hstack((Q_sigmas, Q_rhos))
    
    H = matrix_generator(H_params.flatten(), n_var)
    Q = matrix_generator(Q_params.flatten(), n_var)
    
    Z = np.eye(n_var)
    
    att, Ptt, at, Pt, vt, Ft, Kt, loglike = FastVAR1_kalman_filter(yt, A, Z, H, Q, Lasso=False, lambda_ = .5, Ridge=False, lamda2_=.5)
    
    return -loglike 

def VAR1_obj_fun2(params, yt, Lasso=False, lambda_ = .5, Ridge=False, lamda2_=.5):
    
    n_var = len(yt[0])
    n_corr = int(factorial(n_var)/(2*factorial(n_var-2)))
    
    autoreg_par = params[:n_var]
    H_sigmas = np.exp(params[n_var : n_var + n_var])
    Q_sigmas = np.exp(params[n_var + n_var : n_var + n_var*2])
    H_rhos = sigmoid_like_fun(params[n_var + n_var*2 : n_var + n_var*2 + n_corr])
    Q_rhos = sigmoid_like_fun(params[n_var + n_var*2 + n_corr :])
    
    # A = np.reshape(autoreg_par, (n_var, n_var)) 
    A = np.diag(autoreg_par)
    
    H_params = np.hstack((H_sigmas, H_rhos))
    Q_params = np.hstack((Q_sigmas, Q_rhos))
    
    H = matrix_generator(H_params.flatten(), n_var)
    Q = matrix_generator(Q_params.flatten(), n_var)
    
    Z = np.eye(n_var)
    
    att, Ptt, at, Pt, vt, Ft, Kt, loglike = FastVAR1_kalman_filter(yt, A, Z, H, Q, Lasso, lambda_, Ridge, lamda2_)
    
    return -loglike 

def VAR1_ml_fit(params, yt, model=1, Lasso=False, lambda_ = .5, Ridge=False, lamda2_=.5):
    
    n_var = len(yt[0])
    n_corr = int(factorial(n_var)/(2*factorial(n_var-2)))
    
    if model==1:
        res = minimize(VAR1_obj_fun, params, args=(yt, Lasso, lambda_, Ridge, lamda2_), method='BFGS')
        
        fitted_par = res.x 
        autoreg_par = fitted_par[:n_var*n_var]
        A = np.reshape(autoreg_par, (n_var, n_var))
        
        H_sigmas = np.exp(fitted_par[n_var*n_var : n_var*n_var + n_var])
        Q_sigmas = np.exp(fitted_par[n_var*n_var + n_var : n_var*n_var + n_var*2])
        
        H_rhos = sigmoid_like_fun(fitted_par[n_var*n_var + n_var*2 : n_var*n_var + n_var*2 + n_corr])
        Q_rhos = sigmoid_like_fun(fitted_par[n_var*n_var + n_var*2 + n_corr :])
        
        H_params = np.hstack((H_sigmas, H_rhos))
        Q_params = np.hstack((Q_sigmas, Q_rhos))

        H = matrix_generator(H_params.flatten(), n_var)
        Q = matrix_generator(Q_params.flatten(), n_var)
    
    elif model==2:
        
        res = minimize(VAR1_obj_fun2, params, args=(yt, Lasso, lambda_, Ridge, lamda2_), method='BFGS')
        
        fitted_par = res.x 
        autoreg_par = fitted_par[:n_var]
        A = np.diag(autoreg_par)
        
        H_sigmas = np.exp(fitted_par[n_var : n_var + n_var])
        Q_sigmas = np.exp(fitted_par[n_var + n_var : n_var + n_var*2])
        
        H_rhos = sigmoid_like_fun(fitted_par[n_var + n_var*2 : n_var + n_var*2 + n_corr])
        Q_rhos = sigmoid_like_fun(fitted_par[n_var + n_var*2 + n_corr :])
        
        H_params = np.hstack((H_sigmas, H_rhos))
        Q_params = np.hstack((Q_sigmas, Q_rhos))

        H = matrix_generator(H_params.flatten(), n_var)
        Q = matrix_generator(Q_params.flatten(), n_var)
    
    else:
        raise ValueError('model==1 or model==2')
    
    
    return A, H, Q

def VAR1_em_fit(A0, H0, Q0, yt, maxiter=200, tol=10**-6):
    
    n, n_var = np.shape(yt)
    
    burnIn = n_var + 10
    
    # l = []
        
    A = copy(A0)
    H = copy(H0)
    Q = copy(Q0)
    
    Z = np.eye(n_var)
    
    for i in range(1, maxiter+1):
        
        att, Ptt, at, Pt, vt, Ft, Kt, loglike = VAR1_kalman_filter(yt, A, Z, H, Q)
        x_smooth, V_smooth, Vt_smooth = VAR1Smoothing(yt, A, Z, att, Ptt, Pt, vt, Ft, Kt)
        
        S = np.zeros((n_var, n_var))
        S10 = np.zeros((n_var, n_var))
        eps_smooth = np.zeros((n_var, n_var))
        
        for t in range(burnIn, n):
            
            x0 = np.reshape(x_smooth[t], (n_var, 1))
            x1 = np.reshape(x_smooth[t-1], (n_var, 1))
            
            y = np.reshape(copy(yt[t]), (n_var, 1))

            y = np.reshape(np.copy(yt[t]), (n_var, 1))
            
            S += x0 @ x0.T + V_smooth[t]
            S10 += x0 @ x1.T + Vt_smooth[t]
            
            nan_index = np.isnan(y)
            y[nan_index] = 0
            x0[nan_index] = 0
            
            eps_smooth += (y-x0) @ (y-x0).T + V_smooth[t]
        
        xn = np.reshape(x_smooth[n-1], (n_var, 1))
        x2 = np.reshape(x_smooth[burnIn-1], (n_var, 1))
        
        S00 = S - xn @ xn.T - V_smooth[n-1] + x2 @ x2.T + V_smooth[burnIn-1] 
        S11 = deepcopy(S)
        
        BB = deepcopy(S10)
        AA = deepcopy(S00)
        CC = deepcopy(S11)
        
        H = np.diag(np.diag(eps_smooth)) / (n-burnIn+1)
        
        # diagH = np.diag(H)
        
        # H = eps_smooth / (n-burnIn+1)
        
        Q = (CC - BB - BB.T + AA) / (n-burnIn+1)
        
        diagQ = np.diag(np.diag(Q))
        tempQ = Q - diagQ
        tempQ = .5 * (tempQ + tempQ.T)
        Q = tempQ + diagQ
        
        A = BB @ np.linalg.inv(AA)
    
    # return att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, vt, Ft, A, H, Q 
    return att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, loglike, vt, Ft, Kt, A, H, Q


######## Fitting functions for the Random Walk model with drift 

def RW_kalman_filter(yt, drift, H, Q, burnIn = 0):
    # Kalman filetr for mutivariate random walk with drift
    n_var = len(yt[0]) # number of variables
    N = len(yt)
    
    A = np.eye(n_var)
    Z = np.eye(n_var)
    
    Pt = np.empty((N, n_var, n_var))
    Pt[0] = np.eye(len(Q))
    
    F0 = Z @ Pt[0] @ Z.T + H # Initial valuea of the first innovartion variance
    n_rows, n_cols = np.shape(F0)
    
    Ft = np.empty((N, n_rows, n_cols))
    Ft[0] = F0
    
    
    # Ptt = [] # filtered estimator of the variance of the unobserved process
    # Ptt0 = Pt[0] - Pt[0] @ Z.T @ inv(Ft[0]) @ Z @ Pt[0]
    ZP = Z @ Pt[0]
    Ptt0 = Pt[0] - Pt[0] @ Z.T @ np.linalg.solve(Ft[0], ZP)
    n_rows, n_cols = np.shape(Ptt0)
    
    Ptt = np.empty((N, n_rows, n_cols))
    Ptt[0] = Ptt0
    
    # Kalman gain part
    # K0 = A @ Pt[0] @ Z.T @ inv(Ft[0]) # initial value of the Kalman gain
    tmp = A @ Pt[0] @ Z.T
    K0 = np.linalg.solve(Ft[0].T, tmp.T).T
    n_rows, n_cols = np.shape(K0)
    
    Kt = np.empty((N, n_rows, n_cols))
    Kt[0] = K0
    
    # Allocating the memory for at, att, vt
    
    a0 = copy(yt[0,:]) # initial value for the conditional mean of the unobserved process
    a0 = np.reshape(a0, (1, n_var))
    at = np.empty((N, n_var)) # Conditional mean of the unobserved process
    at[0] = a0
    
    vt = np.empty((N, n_var)) # innovation error
    vt0 = yt[0] - a0  # initial value of the innovation error
    
    nan_index = np.isnan(vt0)
    
    if True in nan_index:
        
        vt0[nan_index]=0
    
    vt[0] = vt0
    
    att = np.empty((N, n_var)) # filtered estimator of the unobserved process
    # att[0,:] = at[0] + Pt[0] @ Z @ inv(Ft[0]) @ vt[0] # computing the initial value of the filtered estimator of the unobserved process
    tmp = Pt[0] @ Z.T
    att[0, :] = at[0] + tmp @ np.linalg.solve(Ft[0], vt[0])
    
    for i in np.arange(1, N):
        
        nan_index = np.isnan(yt[i])
        
        if True in nan_index:
            
            y = deepcopy(yt[i])
            y[nan_index] = 0
            
            Z = deepcopy(Z)
            
            for j in range(0, len(Z)):
                
                Z[nan_index] = 0
                
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3):
                    Pt[i] = copy(Pt[i-1])
                else:
                    Pt[i] = (A @ Ptt[i-1] @ A.T + Q) # Computing the variance
            except:
                pass
            
            # Pt[i] = (A @ Ptt[i-1] @ A.T + Q) # Computing the variance
            
            auxH = np.eye(len(H))
            auxH[~nan_index] = H[~nan_index]
            
            Ft[i] = (Z @ Pt[i] @ Z.T + auxH) # computing the variance of the innovation error
            # Ft[i] = (Z@Pt[i]@np.transpose(Z) + H) # computing the variance of the innovation error
            
            F = inv(Ft[i])
                            
            Ptt[i] = (Pt[i] - Pt[i] @ Z.T @ F @ Z @Pt[i]) # computing the filtered variance
            
            at[i] = A @ att[i-1] + drift
            vt[i] = y - Z @ at[i] 
            
            att[i] = (at[i] + Pt[i] @ Z.T @ F @ np.transpose(vt[i])) 
            
            Kt[i] = A @ Pt[i] @ Z.T @ F # Kalman gain
            
        else:
            
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3) and i>2:
                    Pt[i] = copy(Pt[i-1])
                    # Ft[i] = A @ Pt[i] @ A.T + H
                else:
                    Pt[i] = (A @ Ptt[i-1] @ A.T + Q) # Computing the variance
                    # Ft[i] = (Z @ Pt[i] @ Z.T + H) # computing the variance of the innovation error
            except:
                pass
        
            # Pt[i] = (A @ Ptt[i-1] @ A.T + Q) # Computing the variance
            
            Ft[i] = (Z @ Pt[i] @ Z.T + H) # computing the variance of the innovation error
          
            # F = inv(Ft[i])
            
            # Ptt[i] = (Pt[i] - Pt[i] @ Z.T @ F @ Z @ Pt[i]) # computing the filtered variance
            
            tmp = Pt[i] @ Z.T
            sol = np.linalg.solve(Ft[i], Z @ Pt[i])
            Ptt[i] = Pt[i] - tmp @ sol
            
            at[i] = A @ att[i-1] + drift
            vt[i] = yt[i] - Z @ at[i] 
            
            # att[i] = (at[i] + Pt[i] @ Z.T @ np.linalg.inv(Ft[i]) @ np.transpose(vt[i])) 
            # Solve Ft[i] * x = vt[i].T
            tmp = np.linalg.solve(Ft[i], vt[i].T)
            
            # Then multiply
            att[i] = at[i] + Pt[i] @ Z.T @ tmp
            
            # Kt[i] = A @ Pt[i] @ Z.T @ np.linalg.inv(Ft[i]) # Kalman gain 
            tmp = Z @ Pt[i].T @ A.T
            Kt[i] = np.linalg.solve(Ft[i], tmp.T).T
    
    loglike = 0
    n_var = len(yt[0])
    
    for i in range(burnIn, len(vt)):
        
        loglike += multivariate_gaussian_log_likelihood(vt[i], np.zeros(n_var), Ft[i])
    
    return att, Ptt, at, Pt, vt, Ft, Kt, loglike


@njit
def FastRW_kalman_filter(yt, drift, H, Q, burnIn=0):
    # Numba compatible Kalman filetr for mutivariate random walk with drift
    n_var = yt.shape[1]
    N = yt.shape[0]
    
    A = np.eye(n_var)
    Z = np.eye(n_var)
    
    yt = yt.astype(np.float64)
    A = A.astype(np.float64) 
    Z = Z.astype(np.float64)
    H = H.astype(np.float64)
    Q = Q.astype(np.float64)
    drift = drift.astype(np.float64)
    
    Pt = np.zeros((N, n_var, n_var))
    Pt[0] = np.eye(len(Q))
    
    F0 = Z @ Pt[0] @ np.transpose(Z) + H
    Ft = np.zeros((N, F0.shape[0], F0.shape[1]))
    Ft[0] = F0
    
    # Initial filtered covariance
    ZP = Z @ Pt[0]
    Ptt0 = Pt[0] - Pt[0] @ np.transpose(Z) @ np.linalg.solve(Ft[0], ZP)
    Ptt = np.zeros((N, Ptt0.shape[0], Ptt0.shape[1]))
    Ptt[0] = Ptt0
    
    # Initial Kalman gain
    tmp = A @ Pt[0] @ np.transpose(Z)
    K0 = np.linalg.solve(Ft[0].T, tmp.T).T
    Kt = np.zeros((N, K0.shape[0], K0.shape[1]))
    Kt[0] = K0
    
    at = np.zeros((N, n_var))
    att = np.zeros((N, n_var))
    vt = np.zeros((N, n_var))
    
    a0 = yt[0]
    at[0] = a0
    
    vt0 = yt[0] - a0
    for j in range(n_var):
        if np.isnan(vt0[j]):
            vt0[j] = 0.0
    vt[0] = vt0
    
    tmp0 = Pt[0] @ np.transpose(Z)
    att[0] = at[0] + tmp0 @ np.linalg.solve(Ft[0], vt[0])
    
    for i in range(1, N):
        nan_index = np.isnan(yt[i])
    
        if np.any(nan_index):
            y = np.copy(yt[i])
            for j in range(n_var):
                if nan_index[j]:
                    y[j] = 0.0
    
            Z_mod = np.copy(Z)
            for j in range(Z.shape[0]):
                for k in range(Z.shape[1]):
                    if nan_index[j]:
                        Z_mod[j, k] = 0.0
    
            if i >= 2:
                m = np.mean(Pt[i-1] - Pt[i-2])
                if abs(m) < 1e-3:
                    Pt[i] = Pt[i-1]
                else:
                    Pt[i] = A @ Ptt[i-1] @ A.T + Q
            else:
                Pt[i] = A @ Ptt[i-1] @ A.T + Q
    
            auxH = np.eye(H.shape[0])
            for j in range(H.shape[0]):
                if not nan_index[j]:
                    auxH[j, j] = H[j, j]
    
            Ft[i] = Z_mod @ Pt[i] @ np.transpose(Z_mod) + auxH
    
            sol = np.linalg.solve(Ft[i], Z_mod @ Pt[i])
            Ptt[i] = Pt[i] - Pt[i] @ np.transpose(Z_mod) @ sol
    
            at[i] = A @ att[i-1]
            vt[i] = y - Z_mod @ at[i]
            att[i] = at[i] + Pt[i] @ np.transpose(Z_mod) @ np.linalg.solve(Ft[i], vt[i])
    
            tmp = Z_mod @ Pt[i].T @ A.T
            Kt[i] = np.linalg.solve(Ft[i], tmp.T).T
    
        else:
            if i >= 2:
                m = np.mean(Pt[i-1] - Pt[i-2])
                if abs(m) < 1e-3:
                    Pt[i] = Pt[i-1]
                else:
                    Pt[i] = A @ Ptt[i-1] @ A.T + Q
            else:
                Pt[i] = A @ Ptt[i-1] @ A.T + Q
    
            Ft[i] = Z @ Pt[i] @ np.transpose(Z) + H
    
            sol = np.linalg.solve(Ft[i], Z @ Pt[i])
            Ptt[i] = Pt[i] - Pt[i] @ np.transpose(Z) @ sol
    
            at[i] = A @ att[i-1]
            vt[i] = yt[i] - Z @ at[i]
            att[i] = at[i] + Pt[i] @ np.transpose(Z) @ np.linalg.solve(Ft[i], vt[i])
    
            tmp = Z @ Pt[i].T @ A.T
            Kt[i] = np.linalg.solve(Ft[i], tmp.T).T
    
    loglike = 0.0
    for i in range(burnIn, N):
        loglike += Fast_multivariate_gaussian_log_likelihood(vt[i], np.zeros(n_var), Ft[i])
    
    return att, Ptt, at, Pt, vt, Ft, Kt, loglike


def RW_obj_fun(params, yt, burnIn=0):
    
    n_var = len(yt[0])
    # n_corr = int(factorial(n_var)/(2*factorial(n_var-2)))
    
    drift = params[:n_var]
    H_sigmas = np.exp(params[n_var : n_var + n_var])
    Q_sigmas = np.exp(params[n_var + n_var : n_var * 3])
    Q_rhos = sigmoid_like_fun(params[n_var * 3 :])
    
    # A = np.reshape(autoreg_par, (n_var, n_var))
    
    H = np.diag(H_sigmas)
    Q_params = np.hstack((Q_sigmas, Q_rhos))
    
    Q = matrix_generator(Q_params.flatten(), n_var)
    
    att, Ptt, at, Pt, vt, Ft, Kt, loglike = FastRW_kalman_filter(yt, drift, H, Q, burnIn)
    
    return -loglike 


def RW_ml_fit(params, yt, burnIn=0):
    
    n_var = len(yt[0])    
    
    res = minimize(RW_obj_fun, params, args=(yt, burnIn), method='BFGS')
    
    fitted_par = res.x 
    drift = fitted_par[:n_var]
    
    H_sigmas = np.exp(fitted_par[n_var : n_var + n_var])
    Q_sigmas = np.exp(fitted_par[n_var + n_var : n_var * 3])
    Q_rhos = sigmoid_like_fun(fitted_par[n_var * 3 :])
    
    # A = np.reshape(autoreg_par, (n_var, n_var))
    
    H = np.diag(H_sigmas)
    Q_params = np.hstack((Q_sigmas, Q_rhos))
    
    Q = matrix_generator(Q_params.flatten(), n_var)
    
    return drift, H, Q