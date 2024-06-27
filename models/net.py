import torch.nn as nn
import torch 

cost =  nn.BCELoss( reduction='sum')

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
            nn.Linear(8, 1 ),
            nn.Sigmoid(),
        )
        self.main_module.to(device)
    def forward(self, x):
        return self.main_module(x)
            

class Model:
    def __init__(self, features, device):
        self.net  = Net(features, device=device)
        self.label = torch.tensor([[0],[1]], device=device, dtype=torch.float32)

    def cost_from_batch(self, features, weight_sm, weight_bsm, device ): 
        weight_bsm = torch.maximum(weight_bsm, torch.tensor(0))
        
        combined_features = torch.cat( [features, features])
        #combined_weight   = torch.cat( [weight_sm  / torch.mean(weight_sm), weight_bsm / torch.mean(weight_bsm)]) 
        combined_weight   = torch.cat( [weight_sm, weight_bsm]) 

        combined_weight = torch.minimum( combined_weight, 1e3*torch.median(combined_weight)) # some regularization :) 

        cost.weight = combined_weight
        return cost( self.net(combined_features).ravel(), self.label.expand(2, weight_sm.shape[0]).ravel())