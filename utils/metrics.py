from scipy.stats import ks_2samp
import numpy as np

def net_eval(out, sm, bsm, threshold=0.5, n_points=200):
    
    sm_hist, bsm_hist = make_hist(out, sm, bsm, n_points)
    
    roc, auc = make_roc(sm_hist, bsm_hist)
    
    a   = (sm[(out <= threshold).flatten()].sum() + bsm[(out >= threshold).flatten()].sum())/(sm.sum() + bsm.sum())
    
    
    return roc, auc, a

def make_hist(out, sm, bsm, n_points):
    
    sort_ind = np.argsort(out, axis=0)
    
    bsm       = bsm[sort_ind]
    sm        = sm[sort_ind]
    bsm_total = bsm.sum()
    sm_total  = sm.sum()
    
    disc = len(bsm)%n_points
    
    if disc == 0:
        bsm = np.sum(np.array_split(bsm, n_points), axis=1)
        sm  = np.sum(np.array_split(sm,  n_points), axis=1)
        
    else:
        bsm = np.concatenate([np.sum(np.array_split(bsm, n_points)[:disc]),
                              np.sum(np.array_split(bsm, n_points)[disc:])])
        sm  = np.concatenate([np.sum(np.array_split(sm, n_points)[:disc]),
                              np.sum(np.array_split(sm, n_points)[disc:])])

    return sm, bsm

def make_roc(sm_hist, bsm_hist):
    
    roc = np.array([np.cumsum(sm_hist), np.cumsum(bsm_hist)]).transpose()
    roc /= roc[-1]
    roc = 1 - roc

    auc = -np.trapz(roc[:,1], x=roc[:,0])
    
    return roc, auc   