#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
from torch.utils.data import DataLoader
import numpy as np 
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from utils.options import handleOptions
from utils.metrics import net_eval

import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

def save_and_plot(net, loss_test, loss_train, label, bsm_name, test):

    os.mkdir(f'{label}')

    torch.save(net, f'{label}/network.p')
    torch.save(net.state_dict(), f'{label}/network_state_dict.p')

    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    
    ax.plot( range(len(loss_test)), loss_train, label="Training dataset")
    ax.plot( range(len(loss_test)), loss_test , label="Testing dataset")
    ax.legend()
    fig.savefig(f'{label}/loss.png')
    plt.clf()
    
    fig, ax = plt.subplots(1, 1, figsize=[12,7])
    
    sm_hist,bins,_  = ax.hist(net(test[:][2]).ravel().detach().cpu().numpy(),
                           weights=test[:][0].detach().cpu().numpy(),
                           bins=100, alpha=0.5, label='SM', density=True)
    bsm_hist,_,_ = ax.hist(net(test[:][2]).ravel().detach().cpu().numpy(),
                           weights=test[:][1].detach().cpu().numpy(),
                           bins=bins, alpha=0.5, label='BSM', density=True)
    ax.set_xlabel('Network Output', fontsize=12)
    ax.legend()
    fig.savefig(f'{label}/net_out.png')
    plt.clf()
    
    roc, auc, a = net_eval(net(test[:][2]), test[:][0], test[:][1])
    
    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    ax.plot(roc[:,0], roc[:,1], label='Network Performance')
    ax.plot([0,1],[0,1], ':', label='Baseline')
    ax.legend()
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    fig.savefig(f'{label}/ROC.png')
    plt.clf()
    
    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    ax.plot(roc[:,0], roc[:,1], label='Network Performance')
    ax.plot([0,1],[0,1], ':', label='Baseline')
    ax.legend()
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/ROC_log.png')
    plt.clf()
    
    plt.close()
    
    f = open(f'{label}/performance.txt','w+')
    f.write(    
        'Area under ROC: ' + str(auc) + '\n' + 
        'Accuracy:       ' + str(a) + '\n'
    )
    f.close()

def main():

    args = handleOptions()

    # Now we decide how (if) we will use the gpu
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        args.device = 'cpu'

    # If we use the cpu we dont use the whole UI (at psi)
    torch.set_num_threads(8)

    # all the stuff below should be configurable in the future
    # we get the model = net + cost function
    from models.net import Model
    model = Model(features = len(args.features.split(",")), device = args.device)

    # now we get the data
    from utils.data import eftDataLoader
    signal_dataset = eftDataLoader( args )
    train, test    = torch.utils.data.random_split( signal_dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    dataloader     = DataLoader(  train  , batch_size=args.batch_size, shuffle=True)


    optimizer = optim.SGD(model.net.parameters(), lr=args.learning_rate, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    loss_train = [model.cost_from_batch(train[:][2] , train[:][0], train[:][1], args.device).item()]
    loss_test  = [model.cost_from_batch(test[:][2] , test[:][0], test[:][1], args.device).item()]
    for epoch in tqdm(range(args.epochs)):
        for i,(sm_weight, bsm_weight, features) in enumerate(dataloader):
            if args.profile and ((epoch == 0) and (i == 0)):
                with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                    with record_function('model_inference'):
                        optimizer.zero_grad()
                        loss = model.cost_from_batch(features, sm_weight, bsm_weight, args.device)
                        loss.backward()
                        optimizer.step()
                text_file = open(f'network_profile.txt', 'w')
                n = text_file.write(prof.key_averages().table(sort_by='cpu_time_total'))
                text_file.close()
            else: 
                optimizer.zero_grad()
                loss = model.cost_from_batch(features, sm_weight, bsm_weight, args.device)
                loss.backward()
                optimizer.step()
        loss_train.append( model.cost_from_batch(train[:][2], train[:][0], train[:][1], args.device).item())
        loss_test .append( model.cost_from_batch(test[:][2] , test[:][0], test[:][1], args.device).item())
        scheduler.step(loss_train[epoch])
        if epoch%50==0: 
            save_and_plot( model.net, loss_test, loss_train, f"epoch_{epoch}", signal_dataset.bsm_name, test)
            
    save_and_plot( model.net, loss_test, loss_train, "last", signal_dataset.bsm_name, test)
    
if __name__=="__main__":
    main()
