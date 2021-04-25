# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:37:42 2021

@author: phumt
"""

def calc_R2(hb, hb_pred):
    numerator = np.sum( (hb - hb_pred)**2)
    denominator = np.sum( (hb - np.mean(hb_pred))**2 )
    return 1 - numerator/denominator


def wave_deconstruct(hb):
    t = np.linspace(0, 2*np.pi, len(hb), endpoint=True)
    n = len(t)
    
    W = np.zeros((n, 5))
    wave_params=None
    R2 = []
    if wave_params==None: # wave_params=None
        # Define params for R_peaks as a tuple
        wave_params = [init_rpeak(hb, t)]
        # Initialize other params with zero
        for i in range(5):
            wave_params.append((0, 0, 0, 0, 0))
            
            wave_params[i] = FMM(hb - W.sum(axis=1))
            W[:, i] = sim_hb(wave_params[i], t)
            
            hb_pred = W.sum(axis=1)
            R2_t = calc_R2(hb, hb_pred)
            R2.append(R2_t)
    
        wave_params[0] = FMM(hb - np.delete(W, 0, axis=1).sum(axis=1))
        W[:, 0] = sim_hb(wave_params[0], t)
        hb_pred = W.sum(axis=1)
        R2_t = calc_R2(hb, hb_pred)
        R2.append(R2_t)
        print(R2)
    # Second round
    for i in range(4):
        wave_params[i+1] = FMM(hb - np.delete(W, i+1, axis=1).sum(axis=1))
        
        W_old = W.copy()
        wave_params_old = wave_params.copy()
        
        W[:, i+1] = sim_hb(wave_params[i+1], t)
        hb_pred = W.sum(axis=1)
        
        R2_t = calc_R2(hb, hb_pred)
        
        if R2_t < R2[-1]:
            W = W_old
            wave_params = wave_params_old
        else:
            R2.append(R2_t)
            
    # third round
    for i in range(5):
        wave_params[i] = FMM(hb - np.delete(W, i, axis=1).sum(axis=1))
        
        W_old = W.copy()
        wave_params_old = wave_params.copy()
        
        W[:, i] = sim_hb(wave_params[i], t)
        hb_pred = W.sum(axis=1)
        
        R2_t = calc_R2(hb, hb_pred)
        
        if R2_t < R2[-1]:
            W = W_old
            wave_params = wave_params_old
        else:
            R2.append(R2_t)
            
    # Fourth round
    for i in range(5):
        wave_params[i] = FMM(hb - np.delete(W, i, axis=1).sum(axis=1))
        
        W_old = W.copy()
        wave_params_old = wave_params.copy()
        
        W[:, i] = sim_hb(wave_params[i], t)
        hb_pred = W.sum(axis=1)
        
        R2_t = calc_R2(hb, hb_pred)
        
        if R2_t < R2[-1]:
            W = W_old
            wave_params = wave_params_old
        else:
            R2.append(R2_t)
    
    return hb_pred, W, wave_params, R2


hb_pred1, W1, wave_params1, R21 = wave_deconstruct(hb1)

hb_pred2, W2, wave_params2, R22 = wave_deconstruct(hb2)


fig, ax = plt.subplots(2, 2)
ax[0,0].plot(hb1)
ax[0,0].plot(hb_pred1)
ax[0,1].plot(W1)

ax[1,0].plot(hb2)
ax[1,0].plot(hb_pred2)
ax[1,1].plot(W2)




print(R21)
print(R22)
