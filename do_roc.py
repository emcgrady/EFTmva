import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.options import rocOptions
from utils.buildLikelihood import full_likelihood
import fnmatch
import os


def main():
    args, directory_name = rocOptions()
    # Now we decide how (if) we will use the gpu
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        args.device = 'cpu'
        
     # If we use the cpu we dont use the whole UI (at psi)
    torch.set_num_threads(8)
    
    # now we get the data
    from utils.data import eftDataLoader
    signal_dataset = eftDataLoader( args )
    train,test  = torch.utils.data.random_split( signal_dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    
    sm_weight,bsm_weight,features = test[:]
    likelihood = full_likelihood(args.likelihood)
    test_point = args.test_point.split('=')
    param_score = likelihood( features, {test_point[0]:float(test_point[1])}).detach().numpy()
    param_score = np.minimum(20,param_score)
    fig, ax = plt.subplots(1, 1, figsize=[14,8])
    parametric_sm,bins,_  = ax.hist(param_score, weights=sm_weight.detach() , bins=200 , alpha=0.5, label="SM" , density=True)
    parametric_bsm,_,_ = ax.hist(param_score, weights=bsm_weight.detach(), bins=bins, alpha=0.5, label="BSM", density=True)
    ax.legend()
    fig.savefig(f"{directory_name}/hist.png")
    fig.clf()
    
    for file in os.listdir(args.signal):
        if fnmatch.fnmatch(file, '*last.p'):
            signal = file
        
    dedicated = torch.load(f'{args.signal}/{signal}', map_location=torch.device('cpu'))
    dedicated_score = dedicated(features).detach().numpy()
    fig, ax = plt.subplots(1, 1, figsize=[14,8])
    dedicated_sm,bins,_  = ax.hist(dedicated_score, weights=sm_weight.detach() , bins=200 , alpha=0.5, label="SM" , density=True)
    dedicated_bsm,_,_ = ax.hist(dedicated_score, weights=bsm_weight.detach(), bins=bins, alpha=0.5, label="BSM", density=True)
    ax.legend()
    fig.savefig(f"{directory_name}/hist_dedicated.png")
    fig.clf()

    
    cum_parametric_sm  = np.cumsum( parametric_sm  )
    cum_parametric_sm = cum_parametric_sm/cum_parametric_sm[-1]
    cum_parametric_bsm = np.cumsum( parametric_bsm )
    cum_parametric_bsm = cum_parametric_bsm/cum_parametric_bsm[-1]

    cum_dedicated_sm  = np.cumsum( dedicated_sm  )
    cum_dedicated_sm = cum_dedicated_sm/cum_dedicated_sm[-1]
    cum_dedicated_bsm = np.cumsum( dedicated_bsm )
    cum_dedicated_bsm = cum_dedicated_bsm/cum_dedicated_bsm[-1]
    
    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    ax.plot( cum_parametric_sm, cum_parametric_bsm, label="Parametric discriminator")
    ax.plot( cum_dedicated_sm, cum_dedicated_bsm, label="Dedicated discriminator")
    ax.plot([0, cum_parametric_sm.max()], [0,cum_parametric_bsm.max()], ':')
    ax.set_title(f'{args.test_point}', fontsize=16)
    ax.set_xlabel('Standard Model', fontsize=14)
    ax.set_ylabel('Beyond Standard Model', fontsize=14)
    ax.legend()
    fig.savefig(f"{directory_name}/roc.png")
    fig.clf()

if __name__=="__main__":
    main()
