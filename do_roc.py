import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.options import rocOptions
from utils.buildLikelihood import full_likelihood
from utils.metrics import net_eval
import os


def main():
    args = rocOptions()
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
    
    dedicated = torch.load(f'{args.dedicated}', map_location=torch.device(args.device))
    dedicated_score = dedicated(features).detach().numpy()
    fig, ax = plt.subplots(1, 1, figsize=[14,8])
    dedicated_sm,bins,_  = ax.hist(dedicated_score, weights=sm_weight.detach().numpy(),
                                   bins=200 , alpha=0.5, label="SM" , density=True)
    dedicated_bsm,_   ,_ = ax.hist(dedicated_score, weights=bsm_weight.detach().numpy(),
                                   bins=bins, alpha=0.5, label="BSM", density=True)
    ax.legend()
    fig.savefig(f"{args.name}/hist_dedicated.png")
    fig.clf()
    
    likelihood = full_likelihood(args.likelihood)
    bsm_point = args.bsm_point.replace("=","_").replace(":","_").split('_')
    bsm_point = dict(map(lambda i: (bsm_point[i], float(bsm_point[i+1])), range(len(bsm_point)-1)[::2]))
    param_score = likelihood( features, bsm_point)
    param_score = torch.maximum(torch.tensor(-1000),torch.minimum(torch.tensor(1000),param_score))
    param_score = 1-param_score
    parametric_sm,bins,_  = ax.hist(param_score.detach().numpy(), weights=sm_weight.detach().numpy(), 
                                 bins=200 , alpha=0.5, label="SM" , density=True)
    parametric_bsm,_,_ = ax.hist(param_score.detach().numpy(), weights=bsm_weight.detach().numpy(),
                                 bins=bins, alpha=0.5, label="BSM", density=True)
    ax.legend()
    fig.savefig(f"{args.name}/hist.png")
    fig.clf()    
    
    roc_para, auc_para, a_para = net_eval(param_score, sm_weight, bsm_weight)
    roc_dedi, auc_dedi, a_dedi = net_eval(dedicated(features), sm_weight, bsm_weight)

    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    ax.plot( roc_para[:,0], roc_para[:,1], label="Parametric discriminator")
    ax.plot( roc_dedi[:,0], roc_dedi[:,1], label="Dedicated discriminator")
    ax.plot([0, 1], [0,1], ':')
    ax.set_title(f'{args.bsm_point}', fontsize=16)
    ax.set_xlabel('Standard Model', fontsize=14)
    ax.set_ylabel('Beyond Standard Model', fontsize=14)
    ax.legend()
    fig.savefig(f"{args.name}/roc.png")
    fig.clf()
    
    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    ax.plot( roc_para[:,0], roc_para[:,1], label="Parametric discriminator")
    ax.plot( roc_dedi[:,0], roc_dedi[:,1], label="Dedicated discriminator")
    ax.plot([0, 1], [0,1], ':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'{args.bsm_point}', fontsize=16)
    ax.set_xlabel('Standard Model', fontsize=14)
    ax.set_ylabel('Beyond Standard Model', fontsize=14)
    ax.legend()
    fig.savefig(f"{args.name}/roc_log.png")
    fig.clf()
    
    print(f'Accuracy diff: {a_dedi - a_para}')

if __name__=="__main__":
    main()
