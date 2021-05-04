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
    M : scalar
        Intercept term.
    A : scalar
       Amplitude of the waves. The value 
        is zero if the wave does not exist.
    beta : scalar
        Skewness
    omega : scalar
        Kurtosis
    alpha : scalar
        Mean or the location of each wave
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
            
            if gamma == 0:
                print('Getting zero at a = ', a, ' b = ', b)
                break
            
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

    print(RSS_aux, RSS, rest1, rest2)
    init_params = [M_hat, A_hat, beta_hat, omega_hat, alpha_hat]
    result = minimize(objectiveFunc, init_params, method='nelder-mead')
    
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    print('Solution: %s' % result['x'])

    return result['x']


def sim_hb(params, t):
    '''Inputs must be a list of M_hat, A_hat, beta_hat, 
    omega_hat, and alpha_hat in order. Since the output from the
    FMM function is in this order, we can directly use output from the 
    FMM function.
    '''
    return params[0] + params[1] * np.cos(params[2] + 2*np.arctan(params[3] * np.tan(t-params[4])/2))


def init_rpeak(hb, t):
    '''Return parameters (guess) for an R wave'''
    params_1 = (
        0, # M
        np.max(hb), # A
        np.random.uniform(np.pi/2, 5*np.pi/3), # beta between pi/2 and 5*pi/3
        np.random.uniform(0, 0.12), # omega
        np.median(t), # alpha. Initial guess at midpoint of t
        )
    return params_1


#===========================================================
# hb for testing purpose

hb1 =  np.array([-0.044, -0.038, -0.031, -0.025, -0.014,  0.008,  0.044,  0.045,
        0.034,  0.078,  0.052, -0.028, -0.063, -0.066, -0.058, -0.072,
       -0.055, -0.066, -0.04 ,  0.156,  0.344,  0.247,  0.007, -0.081,
       -0.058, -0.073, -0.088, -0.057, -0.094, -0.072, -0.051, -0.088,
       -0.077, -0.077, -0.067, -0.066, -0.055, -0.058, -0.056, -0.036,
       -0.035, -0.011, -0.001, -0.014,  0.041,  0.081,  0.064,  0.124,
        0.151,  0.137,  0.117,  0.071,  0.054,  0.034,  0.001, -0.057,
       -0.071, -0.057, -0.075, -0.071])

hb2 = np.array([-0.121, -0.09 , -0.076, -0.03 , -0.015, -0.016,  0.033,  0.033,
        0.015,  0.007, -0.021, -0.049, -0.095, -0.081, -0.069, -0.116,
       -0.131, -0.079,  0.136,  0.676,  0.932,  0.273, -0.166, -0.041,
       -0.036, -0.031, -0.038, -0.043, -0.039, -0.052, -0.051, -0.057,
       -0.058, -0.057, -0.042, -0.049, -0.059, -0.037,  0.   ,  0.017,
        0.018,  0.018,  0.02 ,  0.051,  0.072,  0.083,  0.126,  0.155,
        0.172,  0.199,  0.234,  0.259,  0.252,  0.225,  0.197,  0.151,
        0.089,  0.033, -0.019, -0.037])

result = FMM(hb1)

t = np.linspace(0, 2*np.pi, len(hb1), endpoint=True)
hb1_pred = sim_hb(result, t)

calc_R2(hb1, hb1_pred)

fig, ax = plt.subplots()
ax.scatter(np.arange(len(hb1)), hb1)
ax.plot(hb1_pred, color='r')

# 24, 438, 1180 look at the arythmia group

