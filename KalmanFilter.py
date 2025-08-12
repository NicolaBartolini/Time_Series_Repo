# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:41:38 2025

@author: Nicola
"""

import numpy as np
from scipy.optimize import minimize
from copy import deepcopy, copy
from numpy.linalg import inv
from scipy.special import factorial


def multivariate_gaussian_log_likelihood(data, mean, covariance):
    # n, d = data.shape
    n = len(data);
    diff = data - mean

    # Invert the covariance matrix
    inv_covariance = np.linalg.inv(covariance)
    
    log_likelihood = 0;

    # Calculate the log-likelihood
    
    log_likelihood = -0.5 * (n * np.log(2 * np.pi) + np.log(np.linalg.det(covariance)) +
                             diff @ inv_covariance @ np.transpose(diff))

    return log_likelihood

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
    
    


def LeadLagKF(Y, Z_t, T_t, Ht, Qt, burnIn=0):
    
    a0 = np.hstack((Y[0,:], Y[0,:]));
    
    nan_index = np.isnan(a0);
    
    if True in nan_index:
        
        a0[nan_index]=0;
    
    P0 = np.eye(len(Qt));
    
    at = np.empty((len(Y[:,0]), len(a0))); # Conditional mean of the unobserved process
    # Pt = [P0]; # List of conditional variance of the unobserved process
    
    N = len(Y);
    M = np.shape(P0)[0]; 
    K = np.shape(P0)[1];
    
    Pt = np.empty((N, M, K));
    Pt[0] = P0;
    
    att = np.empty((len(Y[:,0]), len(a0) )); # filtered estimator of the unobserved process
    
    at[0] = a0; # initial value for "at"
    
    vt = np.empty((len(Y[:,0]), len(Y[0]))); # innovation error
    vt0 = np.hstack((Y[0,:] )) - a0[:-int(len(a0)/2)]; # initial value of the innovation error
    
    nan_index = np.isnan(vt0);
    
    if True in nan_index:
        
        vt0[nan_index]=0;
    
    vt[0] = vt0;
    
    F0 = Z_t@P0@np.transpose(Z_t) + Ht;
    # Ft = []; # list of variance of the innovation error
    # Ft.append(Z_t@P0@np.transpose(Z_t) + Ht); # the initial value of the first innovation error
    N = len(Y);
    M = np.shape(F0)[0]; 
    K = np.shape(F0)[1];
    
    Ft = np.empty((N, M, K));
    Ft[0] = F0;
    
    # Ptt = []; # filtered estimator of the variance of the unobserved process
    Ptt0 = P0 - P0@np.transpose(Z_t)@np.linalg.inv(Ft[0])@Z_t@P0;
    Ptt = np.empty((len(Y), len(Ptt0), len(Ptt0[0]) ))
    Ptt[0] = Ptt0;
    # Ptt.append(P0- P0@np.transpose(Z_t)@np.linalg.inv(Ft[0])@Z_t@P0); # computing the initial value of the filtered estimator of the variance of the unobserved process
    
    att[0,:] = at[0] + P0@np.transpose(Z_t)@np.linalg.inv(Ft[0])@vt[0]; # computing the initial value of the filtered estimator of the unobserved process
    
    K0 = T_t@Pt[0]@np.transpose(Z_t)@np.linalg.inv(Ft[0])
    
    N = len(Y);
    M = np.shape(K0)[0]; 
    K = np.shape(K0)[1];
    
    Kt = np.empty((N, M, K));
    Kt[0] = K0;
    
    
    for i in np.arange(1, len(Y[:,0])):
        
        nan_index = np.isnan(Y[i]);
        
        if True in nan_index:
            
            y = deepcopy(Y[i]);
            y[nan_index] = 0;
            
            Z = deepcopy(Z_t);
            
            for j in range(0, len(Z_t)):
                
                Z[nan_index] = 0;
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3) and i>2:
                    Pt[i] = copy(Pt[i-1])
                    # Ft[i] = A @ Pt[i] @ A.T + H
                else:
                    Pt[i] = (T_t@Ptt[i-1]@np.transpose(T_t) + Qt); # Computing the variance
                    # Ft[i] = (Z @ Pt[i] @ Z.T + H); # computing the variance of the innovation error
            except:
                pass
            
            # Pt[i] = (T_t@Ptt[i-1]@np.transpose(T_t) + Qt); # Computing the variance
            
            auxH = np.eye(len(Ht));
            auxH[~nan_index] = Ht[~nan_index];
            
            Ft[i] = (Z@Pt[i]@np.transpose(Z) + auxH); # computing the variance of the innovation error
            # Ft[i] = (Z@Pt[i]@np.transpose(Z) + Ht); # computing the variance of the innovation error
            
            Ptt[i] = (Pt[i] - Pt[i]@np.transpose(Z)@np.linalg.inv(Ft[i])@Z@Pt[i]); # computing the filtered variance
            
            at[i] = T_t@att[i-1];
            vt[i] = y - Z@at[i];
            
            att[i] = (at[i] + Pt[i]@np.transpose(Z)@np.linalg.inv(Ft[i])@np.transpose(vt[i]));
            
            Kt[i] = T_t@Pt[i]@np.transpose(Z_t)@np.linalg.inv(Ft[i]); # Kalman gain
            
        else:
            
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3) and i>2:
                    Pt[i] = copy(Pt[i-1])
                    # Ft[i] = A @ Pt[i] @ A.T + H
                else:
                    Pt[i] = (T_t@Ptt[i-1]@np.transpose(T_t) + Qt); # Computing the variance
                    # Ft[i] = (Z @ Pt[i] @ Z.T + H); # computing the variance of the innovation error
            except:
                pass
        
            # Pt[i] = (T_t@Ptt[i-1]@np.transpose(T_t) + Qt); # Computing the variance
            
            Ft[i] = (Z_t@Pt[i]@np.transpose(Z_t) + Ht); # computing the variance of the innovation error
            
            Ptt[i] = (Pt[i] - Pt[i]@np.transpose(Z_t)@np.linalg.inv(Ft[i])@Z_t@Pt[i]); # computing the filtered variance
            
            at[i] = T_t@att[i-1];
            vt[i] = Y[i] - Z_t@at[i];
            
            att[i] = (at[i] + Pt[i]@np.transpose(Z_t)@np.linalg.inv(Ft[i])@np.transpose(vt[i]));
            
            Kt[i] = T_t@Pt[i]@np.transpose(Z_t)@np.linalg.inv(Ft[i]); # Kalman gain
            
    # smoothing
    
    x_smooth = np.empty(np.shape(att));
    x_smooth[-1] = att[-1];
    
    Jtn_0 = Ptt[-1]@ np.transpose(T_t) @np.linalg.inv(Pt[-1]);
    
    N = len(Y);
    M = np.shape(Jtn_0)[0]; 
    K = np.shape(Jtn_0)[1];
    
    Jtn = np.empty((N-1, M, K));
    Jtn[-1] = Jtn_0;
    
    x_smooth[-2]= att[-2]+Jtn[-1]@(x_smooth[-1]-T_t@att[-2]);
    
    # The aoutocorrelation = Vt_smooth
    Vt_smooth_n = (np.eye(len(att[0])) - Kt[-1]@Z_t) @ T_t @ Ptt[-2];
    
    M = np.shape(Vt_smooth_n)[0]; 
    K = np.shape(Vt_smooth_n)[1];
    
    Vt_smooth = np.empty((N,M,K));
    Vt_smooth[-1] = Vt_smooth_n;
    
    # Vt
    
    V_smooth_n = Ptt[-1];
    V_smooth_n1 = Ptt[-2] + Jtn[-1]@(V_smooth_n - Pt[-1])@Jtn[-1];
    
    M = np.shape(V_smooth_n)[0]; 
    K = np.shape(V_smooth_n)[1];
    
    V_smooth = np.empty((N,M,K));
    V_smooth[-1] = V_smooth_n;
    V_smooth[-2] = V_smooth_n1;
    
    
    for i in range(N-2,0,-1):
        
        Jtn[i-1] = Ptt[i-1]@ np.transpose(T_t) @np.linalg.inv(Pt[i]);
        x_smooth[i-1]= att[i-1]+Jtn[i-1]@(x_smooth[i]-T_t@att[i-1]);
        V_smooth[i-1] = Ptt[i-1] + Jtn[i-1]@(V_smooth[i] - Pt[i])@Jtn[i-1];
        Vt_smooth[i] = Ptt[i] @ np.transpose(Jtn[i-1]) + Jtn[i] @ (Vt_smooth[i+1] - T_t@Ptt[i])@np.transpose(Jtn[i-1]);
    
    loglike = 0;
    n_var = len(Y[0]);
    
    for i in range(burnIn, len(vt)):
        
        loglike += multivariate_gaussian_log_likelihood(vt[i], np.zeros(n_var), Ft[i]);
    
    return att.T, Ptt.T, at.T, Pt.T, x_smooth.T, V_smooth.T, Vt_smooth.T, vt, Ft, loglike;



def LeadLagllem(y, Q_init, R_init, F_init, C, maxiter=3000, eps=10**-4):
    
    # Matrix d*T d=number of variables, T=number of observations; 
    
    # Z_t = C
    
    # Q_init = Q_t;
    # R_init = R_t;
    # F_init = T_t;
    # C = Z_t;
    
    Q = Q_init;
    R = R_init;
    F = F_init;
    
    [d, T] = np.shape(y);
    
    Id = np.eye(d);
    
    P = np.hstack((Id, np.zeros((d,d))));
    
    burnIn = d + 10;
    
    l = np.ones(maxiter+1)*(10**10);
    # delta_log = np.ones(maxiter+1)*(10^10);
    
    l_old = 10**10;
    
    ###
    for i in range(1, maxiter+1):
    
        S = np.zeros((2*d, 2*d));
        S10 = np.zeros((2*d, 2*d));
        eps_smooth = np.zeros((d, d));
        
        # kalman_filter(Y, Z_t, T_t, Ht, Q_t)
        
        att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, vt, Ft, loglike = LeadLagKF(y.T, C, F_init, R, Q, burnIn);
        # att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, vt, Ft, loglike = kalman_filter(y.T, C, F, R, Q, burnIn);
        
        for t in range(burnIn, T):
            
            l[i] = loglike;
            l_old = abs(l[i-1] - loglike)
            
            if True in np.isnan(y[:,t]):
                
                nan_index = np.where(np.isnan(y[:,t]));
                no_nan_index = np.where(~np.isnan(y[:,t]));
                
                y1 = np.zeros(np.shape(copy(y[:,t])));
                y1[no_nan_index] = copy(y[:,t][no_nan_index]);
                y1 = np.reshape(y1, (len(y1),1));
                
                x = x_smooth[:,t];
                x = np.reshape(x, (len(x),1));
                
                x1 = x_smooth[:,t-1];
                x1 = np.reshape(x1, (len(x1),1));
                
                S = S + x @ x.T + V_smooth[:,:,t];
                S10 = S10 + x1 @ x1.T + Vt_smooth[:,:,t];
                
                auxY = np.copy(y1);
                auxY = copy(y1);
                
                auxC = np.zeros(np.shape(C));
                auxC[no_nan_index] = copy(C[no_nan_index]);
                
                R2 = copy(R);
                R2[no_nan_index] = 0;
                
                eps_smooth += (auxY - auxC@x) @ (auxY - auxC@x).T + auxC@V_smooth[:,:,t]@auxC.T + R2;
                
            else:
                
                x = x_smooth[:,t];
                x = np.reshape(x, (len(x),1));
                
                x1 = x_smooth[:,t-1];
                x1 = np.reshape(x1, (len(x1),1));
                
                S = S + x @ x.T + V_smooth[:,:,t];
                S10 = S10 + x1 @ x1.T + Vt_smooth[:,:,t];
                
                auxY = np.reshape(y[:,t], (len(y[:,t]),1));
                
                eps_smooth += (auxY - C@x) @ (auxY - C@x).T + C@V_smooth[:,:,t]@C.T;
        
        x = x_smooth[:,T-1];
        x = np.reshape(x, (len(x),1));
        
        x1 = x_smooth[:,burnIn-1];
        x1 = np.reshape(x1, (len(x1),1));
        
        S00 = S - x@x.T - V_smooth[:,:,T-1] + x1 @ x1.T + V_smooth[:,:,burnIn-1];
        S11 = deepcopy(S);
        
        BB = deepcopy(S10);
        AA = deepcopy(S00);
        CC = deepcopy(S11);
        
        BB1 = BB[0:d, 0:d];
        BB2 = BB[0:d, d:];
        BBtilde = np.hstack((BB1, BB2));
        
        AA11 = AA[0:d, 0:d];
        AA12 = AA[0:d, d:];
        AA21 = AA[d:, 0:d];
        AA22 = AA[d:, d:];
        
        Gamma = BB1 - BB2 - AA11 + AA12;
        Theta = AA11 + AA22 - AA12 - AA21;
        
        auxF = Gamma @ np.linalg.inv(Theta);
        Ftilde = np.hstack((Id + auxF, -auxF));
        
        auxQ = P@CC@P.T - BBtilde@Ftilde.T - Ftilde@BBtilde.T + Ftilde@AA@Ftilde.T
        auxQ = auxQ/(T-burnIn+1);
        
        temp_Q = np.tril(auxQ) + np.tril(auxQ, -1).T;
        Q = np.hstack((temp_Q, np.zeros((d,d))));
        Q = np.vstack((Q, np.zeros((d, 2*d))));
        
        Fgiu = np.hstack((Id, np.zeros((d,d))));
        
        F = np.vstack((Ftilde, Fgiu));
        
        temp_diag_R = np.diag(eps_smooth/(T-burnIn+1));
        # print(temp_diag_R)
        # print(np.where(np.isnan(temp_diag_R)))
        
        if True in np.isnan(temp_diag_R):
            temp_diag_R[np.where(np.isnan(temp_diag_R))] = 1;
        elif 0 in temp_diag_R:
            temp_diag_R[temp_diag_R==0] = 1;
        
        R = np.diag(temp_diag_R);
        
        
        if l_old<=eps:
            print('Tolerance level: '+str(l_old));
            print('N. of iteration: '+str(i));
            break;
            
        if i==maxiter:
            print('Tolerance level: '+str(l_old));
            print('N. of iteration: '+str(i));
    
    return F, R, Q, att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, vt, Ft, loglike;


def LeadLagObjFun(params, yt):
    
    n_var = len(yt[0])
    # n_corr = int(factorial(n_var)/(2*factorial(n_var-2)))
    
    leadlag_par = params[:n_var*n_var]
    H_sigmas = np.exp(params[n_var*n_var : n_var*n_var + n_var])
    Q_sigmas = np.exp(params[n_var*n_var + n_var : n_var*n_var + n_var*2])
    Q_rhos = sigmoid_like_fun(params[n_var*n_var + n_var*2 : ])
    
    ####### The lead lag matrix
    
    F = np.empty((n_var, n_var))
    
    for i in range(0, n_var):
        for j in range(0, n_var):
            F[i,j] = leadlag_par[i*j + i] 
    
    Tsu = np.hstack((np.eye(n_var) + F, -F));
    Tgiu = np.hstack((np.eye(n_var), np.zeros((n_var, n_var))));
    T_t = np.vstack((Tsu, Tgiu)); # Final lead-lag matrix (Phi in the paper/thesis)
    
    # H_params = np.hstack((H_sigmas, H_rhos))
    Q_params = np.hstack((Q_sigmas, Q_rhos))
    
    H = np.diag(H_sigmas)
    Q = matrix_generator(Q_params.flatten(), n_var)
    
    Q_t = np.hstack((Q, np.zeros(np.shape(Q))));
    Q_t = np.vstack((Q_t, np.zeros((len(Q_t),len(Q_t[0]*2)))));
    
    Z_t = np.hstack((np.eye(n_var), np.zeros((n_var, n_var))))
    # print(H)
    
    att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, vt, Ft, loglike = LeadLagKF(yt, Z_t, T_t, H, Q_t, burnIn=10)
    
    return -loglike

def LeadLagMLfit(params, yt, n_var):
    
    res = minimize(LeadLagObjFun, params, args=(yt), method='BFGS')
    
    fitted_params = res.x 
    
    leadlag_par = fitted_params[:n_var*n_var]
    H_sigmas = np.exp(fitted_params[n_var*n_var : n_var*n_var + n_var])
    Q_sigmas = np.exp(fitted_params[n_var*n_var + n_var : n_var*n_var + n_var*2])
    Q_rhos = sigmoid_like_fun(fitted_params[n_var*n_var + n_var*2 : ])
    
    ####### The lead lag matrix
    
    F = np.empty((n_var, n_var))
    
    for i in range(0, n_var):
        for j in range(0, n_var):
            F[i,j] = leadlag_par[i*j + i] 
    
    Tsu = np.hstack((np.eye(n_var) + F, -F));
    Tgiu = np.hstack((np.eye(n_var), np.zeros((n_var, n_var))));
    T_t = np.vstack((Tsu, Tgiu)); # Final lead-lag matrix (Phi in the paper/thesis)
    
    # H_params = np.hstack((H_sigmas, H_rhos))
    Q_params = np.hstack((Q_sigmas, Q_rhos))
    
    H = np.diag(H_sigmas)
    Q = matrix_generator(Q_params.flatten(), n_var)
    
    return F, H, Q, T_t

def VAR1_kalman_filter(yt, A, Z, H, Q):
    
    n_var = len(yt[0]) # number of variables
    N = len(yt)
    
    Pt = np.empty((N, n_var, n_var))
    Pt[0] = np.eye(len(Q))
    
    F0 = Z @ Pt[0] @ Z.T + H; # Initial valuea of the first innovartion variance
    n_rows, n_cols = np.shape(F0)
    
    Ft = np.empty((N, n_rows, n_cols))
    Ft[0] = F0
    
    
    # Ptt = []; # filtered estimator of the variance of the unobserved process
    Ptt0 = Pt[0] - Pt[0] @ Z.T @ inv(Ft[0]) @ Z @ Pt[0]
    n_rows, n_cols = np.shape(Ptt0)
    
    Ptt = np.empty((N, n_rows, n_cols))
    Ptt[0] = Ptt0;
    
    # Kalman gain part
    K0 = A @ Pt[0] @ Z.T @ inv(Ft[0]); # initial value of the Kalman gain
    n_rows, n_cols = np.shape(K0)
    
    Kt = np.empty((N, n_rows, n_cols))
    Kt[0] = K0
    
    # Allocating the memory for at, att, vt
    
    a0 = copy(yt[0,:]); # initial value for the conditional mean of the unobserved process
    a0 = np.reshape(a0, (1, n_var))
    at = np.empty((N, n_var)); # Conditional mean of the unobserved process
    at[0] = a0
    
    vt = np.empty((N, n_var)); # innovation error
    vt0 = yt[0] - a0; # initial value of the innovation error
    
    nan_index = np.isnan(vt0);
    
    if True in nan_index:
        
        vt0[nan_index]=0;
    
    vt[0] = vt0
    
    att = np.empty((N, n_var)); # filtered estimator of the unobserved process
    att[0,:] = at[0] + Pt[0] @ Z @ inv(Ft[0]) @ vt[0]; # computing the initial value of the filtered estimator of the unobserved process
    
    for i in np.arange(1, N):
        
        nan_index = np.isnan(yt[i]);
        
        if True in nan_index:
            
            y = deepcopy(yt[i]);
            y[nan_index] = 0;
            
            Z = deepcopy(Z);
            
            for j in range(0, len(Z)):
                
                Z[nan_index] = 0;
                
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3):
                    Pt[i] = copy(Pt[i-1])
                else:
                    Pt[i] = (A @ Ptt[i-1] @ A.T + Q); # Computing the variance
            except:
                pass
            
            # Pt[i] = (A @ Ptt[i-1] @ A.T + Q); # Computing the variance
            
            auxH = np.eye(len(H));
            auxH[~nan_index] = H[~nan_index];
            
            Ft[i] = (Z @ Pt[i] @ Z.T + auxH); # computing the variance of the innovation error
            # Ft[i] = (Z@Pt[i]@np.transpose(Z) + H); # computing the variance of the innovation error
            
            F = inv(Ft[i])
                            
            Ptt[i] = (Pt[i] - Pt[i] @ Z.T @ F @ Z @Pt[i]); # computing the filtered variance
            
            at[i] = A @ att[i-1];
            vt[i] = y - Z @ at[i];
            
            att[i] = (at[i] + Pt[i] @ Z.T @ F @ np.transpose(vt[i]));
            
            Kt[i] = A @ Pt[i] @ Z.T @ F; # Kalman gain
            
        else:
            
            try:
                m = np.mean(Pt[i-1]-Pt[i-2])
                if abs(m)<10**(-3) and i>2:
                    Pt[i] = copy(Pt[i-1])
                    # Ft[i] = A @ Pt[i] @ A.T + H
                else:
                    Pt[i] = (A @ Ptt[i-1] @ A.T + Q); # Computing the variance
                    # Ft[i] = (Z @ Pt[i] @ Z.T + H); # computing the variance of the innovation error
            except:
                pass
        
            # Pt[i] = (A @ Ptt[i-1] @ A.T + Q); # Computing the variance
            
            Ft[i] = (Z @ Pt[i] @ Z.T + H); # computing the variance of the innovation error
          
            F = inv(Ft[i])
            
            Ptt[i] = (Pt[i] - Pt[i] @ Z.T @ F @ Z @ Pt[i]); # computing the filtered variance
            
            at[i] = A @ att[i-1];
            vt[i] = yt[i] - Z @ at[i];
            
            att[i] = (at[i] + Pt[i] @ Z.T @ F @ np.transpose(vt[i]));
            
            Kt[i] = A @ Pt[i] @ Z.T @ F; # Kalman gain
        
    # Smoothing
    
    x_smooth = np.empty(np.shape(att));
    x_smooth[-1] = att[-1];
    
    Jtn_0 = Ptt[-1]@ np.transpose(A) @np.linalg.inv(Pt[-1]);
    
    n_rows, n_cols = np.shape(Jtn_0); 
    
    Jtn = np.empty((N-1, n_rows, n_cols));
    Jtn[-1] = Jtn_0;
    
    x_smooth[-2]= att[-2] + Jtn[-1] @ (x_smooth[-1] - A @ att[-2]);
    
    # The aoutocorrelation = Vt_smooth
    Vt_smooth_n = (np.eye(len(att[0])) - Kt[-1] @ Z) @ A @ Ptt[-2];
    
    n_rows, n_cols = np.shape(Vt_smooth_n); 
    
    Vt_smooth = np.empty((N, n_rows, n_cols));
    Vt_smooth[-1] = Vt_smooth_n;
    
    # Vt
    V_smooth_n = Ptt[-1];
    V_smooth_n1 = Ptt[-2] + Jtn[-1] @ (V_smooth_n - Pt[-1]) @ Jtn[-1];
    
    n_rows, n_cols = np.shape(V_smooth_n); 
    
    V_smooth = np.empty((N, n_rows, n_cols));
    V_smooth[-1] = V_smooth_n;
    V_smooth[-2] = V_smooth_n1;
    
    
    for i in range(N-2,0,-1):
        
        Jtn[i-1] = Ptt[i-1]@ np.transpose(A) @np.linalg.inv(Pt[i]);
        x_smooth[i-1]= att[i-1]+Jtn[i-1]@(x_smooth[i]-A@att[i-1]);
        V_smooth[i-1] = Ptt[i-1] + Jtn[i-1]@(V_smooth[i] - Pt[i])@Jtn[i-1];
        Vt_smooth[i] = Ptt[i] @ np.transpose(Jtn[i-1]) + Jtn[i] @ (Vt_smooth[i+1] - A@Ptt[i])@np.transpose(Jtn[i-1]);
    
    loglike = 0;
    
    burnIn = 0
    
    for i in range(burnIn, N ):
        
        loglike += multivariate_gaussian_log_likelihood(vt[i], np.zeros(n_var), Ft[i]);
    
    return att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, loglike, vt, Ft


def VAR1_obj_fun(params, yt):
    
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
    
    # print(H)
    # print(Q)
    # print()
    
    att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, loglike, vt, Ft = VAR1_kalman_filter(yt, A, Z, H, Q)
    
    return -loglike

def VAR1_ml_fit(params, yt):
    
    n_var = len(yt[0])
    n_corr = int(factorial(n_var)/(2*factorial(n_var-2)))
    
    res = minimize(VAR1_obj_fun, params, args=(yt), method='BFGS')
    
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
    
    return A, H, Q


def VAR1_em_fit(A0, H0, Q0, yt, maxiter=200, tol=10**-6):
    
    n, n_var = np.shape(yt)
    
    burnIn = n_var + 10
    
    l = [];
        
    A = copy(A0)
    H = copy(H0)
    Q = copy(Q0)
    
    Z = np.eye(n_var)
    
    for i in range(1, maxiter+1):
        
        att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, loglike, vt, Ft = VAR1_kalman_filter(yt, A, Z, H, Q)
        
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
            
            # C = np.ones(np.shape(V_smooth[t]))
            # print(C)
            # print(nan_index)
            # C[nan_index] = 0
            
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
        
        if i==1:
            l.append(loglike)
            continue
        
        if abs(l[-1] - loglike)<=tol:
            print('Iterations : '+str(i))
            return att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, loglike, vt, Ft, A, H, Q
    
    # print('Maximun number of iterations reached')
    return att, Ptt, at, Pt, x_smooth, V_smooth, Vt_smooth, loglike, vt, Ft, A, H, Q



    
