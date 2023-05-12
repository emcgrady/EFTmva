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

class full_likelihood:
    def __init__(self, input_file):
        with open(input_file) as f:
            self.configuration = yaml.safe_load( f.read() )

        self.wcs = self.configuration['wcs'].split(",") # coefficients to parametrize the likelihood
        self.quadratic = {}; self.linear={}
        for wc in self.wcs:
            self.quadratic[wc]      = likelihood_term( torch.load( self.configuration[f'{wc}_quad'], map_location=torch.device('cpu')))
            value_forlinear, net_forlinear = self.configuration[f'{wc}_forlinear'].split(",")
            self.linear[wc]      = linear_term(self.quadratic[wc], likelihood_term(torch.load( net_forlinear, map_location=torch.device('cpu'))), value_forlinear)
            

    def __call__(self, features, coef_values):
        if set(self.wcs)!=set(coef_values.keys()):
            print(self.wcs, coef_values.keys())
            raise RuntimeError(f"The coefs passed to the likelihood do not align with those used in the likelihood parametrization")

        likelihood_ratio = torch.ones_like(features[:,0]) # SM/SM, just for sanity
        for wc in self.wcs:
            linear_term = (self.linear[wc](features)*coef_values[wc]).flatten()
            quadratic_term = (self.quadratic[wc](features)*coef_values[wc]*coef_values[wc]).flatten()
            likelihood_ratio = likelihood_ratio + linear_term + quadratic_term


            # to do: crossed terms

        return likelihood_ratio
        
