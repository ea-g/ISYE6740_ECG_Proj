# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

def FMM(hb):
    '''FMM algorithm for a heartbeat signal
    Parameters
    ----------
    t : list
        The indexes of each heart beat. X(t_i) for t_1 < ... < t_n and that
        t_i in [0, 2*pi]
    hbs : list
        Mv values for the heartbeats
        
    Returns
    ----------
    A : 1x5 list
        A 4x1 list containing the amplitude of the waves. The value 
        is zero if the wave does not exist.
    alpha : list
        A 1x5 list containing the means or the locations of each wave
    beta : list
        A 1x5 list containing the skewness of each wave
    omega : list
        A 1x5 list containing the kurtosis of each wave
    '''

    # Step 1: Compute the initial estimates for M, A, alpha, beta, omega
    RSS = 9999999
    n = len(hb)
    t = np.linspace(0, 2*np.pi, len(hb), endpoint=True) # the time step is between 0, 2*pi
    # alpha is between 0 and 2*pi
    alpha = np.linspace(0, 2*np.pi, num=n, endpoint=True)
    # omega is between 0 and 1
    omega = np.linspace(0, 1, num=n, endpoint=True)

    for a in range(len(alpha)):
        for b in range(len(omega)):
            t_star = alpha[a] + 2*np.arctan(omega[b]*np.tan( (t-alpha[a])/2 ) )

            # Fitting a cos-sin regression:
            x_cos = np.cos(t_star)
            x_sin = np.cos(t_star)
            X = np.vstack((x_cos, x_sin)).T
            lm = LinearRegression()
            lm.fit(X, hb.T)
        
            delta = lm.coef_[0] # coefficient of cosine
            gamma = lm.coef_[1] # coefficient of sine
            
            print('X: \n', X)
            print('hb: \n', hb.T)
            print('gamma is: ', gamma)
            
            M_1 = lm.intercept_ # the intercept term
            A_1 = np.sqrt(delta**2 + gamma**2)
            alpha_1 = alpha[a]
            beta_1 = np.arctan(-delta/gamma) + alpha_1
            omega_1 = omega[b]
            
            MobiusReg = M_1 + A_1 * np.cos(beta_1 + 2*np.arctan(omega_1 * np.tan(t-alpha_1)/2))
            RSS_aux = np.sum((hb - MobiusReg)**2)/n
            
            maxi = M_1 + A_1
            mini = M_1 - A_1
            
            s = np.sqrt(RSS/5)
            rest1 = maxi <= (np.max(hb) * 1.96*s)
            rest2 = mini >= (np.min(hb) - 1.96*s)
            
            if (RSS_aux < RSS) & rest1 & rest2:
                M_hat = M_1
                A_hat = A_1
                alpha_hat = alpha_1
                beta_hat = beta_1
                omega_hat = omega_1
                RSS = RSS_aux

    # Step 2: Conduct a Nelder-Mead optimization method to compute the final estimates
    # From https://machinelearningmastery.com/how-to-use-nelder-mead-optimization-in-python/
    
    # Perform the search
    def objectiveFunc(params):
        J = params[0] + params[1] * np.cos(params[2] + 2*np.arctan(params[3] * np.tan(t-params[4])/2))
        return np.sum( (hb - J)**2/ n )
    
    init_params = [M_hat, A_hat, beta_hat, omega_hat, alpha_hat]
    result = minimize(objectiveFunc, init_params, method='nelder-mead')
    
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    print('Solution: %s' % result['x'])

    return result['x'][1], result['x'][4], result['x'][2], result['x'][3]


def FMM_ecg():
    pass




#===========================================================
# hb for testing purpose
hb =  np.array([-0.044, -0.038, -0.031, -0.025, -0.014,  0.008,  0.044,  0.045,
        0.034,  0.078,  0.052, -0.028, -0.063, -0.066, -0.058, -0.072,
       -0.055, -0.066, -0.04 ,  0.156,  0.344,  0.247,  0.007, -0.081,
       -0.058, -0.073, -0.088, -0.057, -0.094, -0.072, -0.051, -0.088,
       -0.077, -0.077, -0.067, -0.066, -0.055, -0.058, -0.056, -0.036,
       -0.035, -0.011, -0.001, -0.014,  0.041,  0.081,  0.064,  0.124,
        0.151,  0.137,  0.117,  0.071,  0.054,  0.034,  0.001, -0.057,
       -0.071, -0.057, -0.075, -0.071])

FMM(hb)


