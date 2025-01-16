import torch.nn as nn
import torch 

cost =  nn.BCELoss(reduction='sum')

class Net(nn.Module):
    def __init__(self, features, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(features, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.main_module.to(device)
        self.main_module.type(torch.float64)
    def forward(self, x):
        return self.main_module(x)
            

class Model:
    def __init__(self, features, device):
        self.net  = Net(features, device=device)
        self.bkg  = torch.tensor([0], device=device, dtype=torch.float64)
        self.sgnl = torch.tensor([1], device=device, dtype=torch.float64)

    def cost_from_batch(self, features, weight_sm, weight_bsm, sm_mean, bsm_mean, device ):
        half_length = features.size(0) // 2
        cost.weight = torch.cat( [weight_sm[:half_length] /sm_mean, weight_bsm[half_length:]/bsm_mean]) 
        targets     = torch.cat([self.bkg.expand(1, half_length), self.sgnl.expand(1, features.size(0) - half_length)]).ravel()
        
        return cost(self.net(features).ravel(), targets)