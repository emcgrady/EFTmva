import torch

def net_eval(out, sm, bsm, threshold=0.5, n_points=200):
    
    sm_hist, bsm_hist = make_hist(out, sm, bsm, n_points)
    
    roc, auc = make_roc(sm_hist, bsm_hist, n_points)
    
    a   = (sm[(out <= threshold).flatten()].sum() + bsm[(out >= threshold).flatten()].sum())/(sm.sum() + bsm.sum())
    
    
    return roc, auc, a

def make_hist(out, sm, bsm, n_points):
    
    sort_ind = torch.argsort(out, axis=0)
    
    bsm       = bsm[sort_ind]
    sm        = sm[sort_ind]
    bsm_total = bsm.sum()
    sm_total  = sm.sum()
    
    disc = len(bsm)%n_points
    
    if disc == 0:
        bsm = torch.sum(torch.tensor_split(bsm, n_points), axis=1)
        sm  = torch.sum(torch.tensor_split(sm,  n_points), axis=1)
        
    else:
        bsm = torch.cat((torch.sum(torch.stack(torch.tensor_split(bsm, n_points)[:disc]), axis=1),
                         torch.sum(torch.stack(torch.tensor_split(bsm, n_points)[disc:]), axis=1)), axis=0)
        sm  = torch.cat((torch.sum(torch.stack(torch.tensor_split(sm,  n_points)[:disc]), axis=1),
                         torch.sum(torch.stack(torch.tensor_split(sm,  n_points)[disc:]), axis=1)), axis=0)

    return sm, bsm

def make_roc(sm_hist, bsm_hist, n_points):
    
    roc = torch.cat((torch.cumsum(sm_hist, dim=0).reshape(n_points,1),
                     torch.cumsum(bsm_hist, dim=0).reshape(n_points,1)), axis=1)
    roc = 1 - roc/roc[-1]

    auc = -torch.trapz(roc[:,1], x=roc[:,0])
    
    return roc, auc   