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
    def forward(self, x):
        return self.main_module(x)
            

class Model:
    def __init__(self, features, device):
        self.net  = Net(features, device=device)
        self.label = torch.tensor([[0],[1]], device=device, dtype=torch.float64)

    def cost_from_batch(self, features, weight_sm, weight_bsm, sm_mean, bsm_mean, device ): 
        combined_features = torch.cat( [features, features])
        cost.weight   = torch.cat( [weight_sm /sm_mean, weight_bsm/bsm_mean]) 
        
        return cost( self.net(combined_features).ravel(), self.label.expand(2, weight_sm.shape[0]).ravel())