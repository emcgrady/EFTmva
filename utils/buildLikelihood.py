import torch
import yaml 
import numpy as np

class likelihood_term:
    # Use the ALICE method (1808.00973) to estimate the likelihood ratio
    # Needs a net trained with cross entropy as an input
    def __init__(self, input_net):
        self.input_net = input_net
    def __call__(self, features):
        score = self.input_net(features)
        return (1-score)/score

class linear_term:
    # extracts the linear term from the quadratic term and the likelihood ratio evaluated at a given point
    def __init__(self, quad_term, for_linear_term, for_linear_value):
        self.quad_term        = quad_term
        self.for_linear_term  = for_linear_term
        self.for_linear_value = float(for_linear_value)

    def __call__(self, features):
        return self.for_linear_term(features)/self.for_linear_value - self.for_linear_value*self.quad_term(features) - 1 / self.for_linear_value
    
    
class crossed_term:
    def __init__(self, crossed_term, for_linear_term0, for_linear_term1, for_linear_value0, for_linear_value1):
        self.crossed_term      = crossed_term
        self.for_linear_term0  = for_linear_term0
        self.for_linear_term1  = for_linear_term1
        self.for_linear_value0 = float(for_linear_value0)
        self.for_linear_value1 = float(for_linear_value1)
        
    def __call__(self, features):
        return (self.crossed_term(features) - self.for_linear_term0(features) - self.for_linear_term1(features) + 1) / (self.for_linear_value0*self.for_linear_value1)

class full_likelihood:
    def __init__(self, input_file):
        with open(input_file) as f:
            self.configuration = yaml.safe_load( f.read() )

        self.wcs = self.configuration['wcs'].split(",")
        self.quadratic = {}; self.linear={}; value_forlinear={}; net_forlinear={}
        
        for wc in self.wcs:
            self.quadratic[wc] = likelihood_term(torch.load(self.configuration[f'{wc}_quad'], map_location=torch.device('cpu')))
            value_forlinear[wc], net_forlinear[wc] = self.configuration[f'{wc}_forlinear'].split(",")
            self.linear[wc] = linear_term(self.quadratic[wc], likelihood_term(torch.load(net_forlinear[wc])), value_forlinear[wc])
                                                                              
        if len(self.wcs) > 1:
            self.crossed_net = torch.load(self.configuration[f'{self.wcs[0]}_{self.wcs[1]}_comb'])
            self.crossed = crossed_term(likelihood_term(self.crossed_net),
                                        likelihood_term(torch.load(net_forlinear[self.wcs[0]], map_location=torch.device('cpu'))),
                                        likelihood_term(torch.load(net_forlinear[self.wcs[0]], map_location=torch.device('cpu'))),
                                        value_forlinear[self.wcs[0]], value_forlinear[self.wcs[1]])

    def __call__(self, features, coef_values):
        if set(self.wcs)!=set(coef_values.keys()):
            print(self.wcs, coef_values.keys())
            raise RuntimeError(f"The coefs passed to the likelihood do not align with those used in the likelihood parametrization")
        linear_term={}; quadratic_term={}
        likelihood_ratio = torch.ones_like(features[:,0])
        
        for wc in self.wcs:
            linear_term[wc]    = (self.linear[wc](features)*coef_values[wc]).flatten()
            quadratic_term[wc] = (self.quadratic[wc](features)*coef_values[wc]*coef_values[wc]).flatten()
            likelihood_ratio   = likelihood_ratio + linear_term[wc] + quadratic_term[wc]
            
        if len(self.wcs) > 1:
            cross_term       = (self.crossed(features)*coef_values[self.wcs[0]]*coef_values[self.wcs[1]]).flatten()
            likelihood_ratio = likelihood_ratio + cross_term

        return 1/(likelihood_ratio+1)
        
