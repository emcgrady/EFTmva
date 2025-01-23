import torch
import yaml 
import numpy as np

class likelihood_term:
    '''
    Converts the output of a neural network trained with cross entropy to a likelihood ratio
    Initialize with the neural network and the normalization factor used in the training
    '''
    def __init__(self, input_net, norm):
        '''
        input_net: neural network trained with cross entropy
        norm: normalization factor used in the training
        '''
        self.input_net = input_net
        self.norm = norm
    def __call__(self, features):
        '''
        features: input features to the neural network
        '''
        features = features.to(torch.float64)
        score = self.input_net(features)
        return self.norm*score/(1-score)

class linear_term:
    '''
    Returns the linear term of the likelihood ratio given the quadratic term and the full likelihood ratio evaluated at a given point in WC space
    '''
    # extracts the linear term from the quadratic term and the likelihood ratio evaluated at a given point
    def __init__(self, quad_term, for_linear_term, for_linear_value):
        '''
        quad_term: quadratic term of the likelihood ratio
        for_linear_term: full likelihood ratio evaluated at a given point in WC space
        for_linear_value: WC value used to train the full likelihood ratio 
        '''
        self.quad_term        = quad_term
        self.for_linear_term  = for_linear_term
        self.for_linear_value = float(for_linear_value)

    def __call__(self, features):
        '''
        features: input features to the neural network
        '''
        return self.for_linear_term(features)/self.for_linear_value - self.for_linear_value*self.quad_term(features) - 1 / self.for_linear_value
    
    
class crossed_term:
    '''
    Returns the bsm crossed term of the likelihood ratio given two quadratic terms and two full likelihood ratios evaluated at a given points in WC space
    '''
    def __init__(self, crossed_term, for_linear_term0, for_linear_term1, for_linear_value0, for_linear_value1):
        '''
        crossed_term: converted likelihood ratio for network trained at two points in WC space
        for_linear_term0: first converted likelihood ratio evaluated at a given point in WC space
        for_linear_term1: second converted likelihood ratio evaluated at a given point in WC space
        for_linear_value0: WC value used to train the first full likelihood ratio
        for_linear_value1: WC value used to train the second full likelihood ratio
        '''
        self.crossed_term      = crossed_term
        self.for_linear_term0  = for_linear_term0
        self.for_linear_term1  = for_linear_term1
        self.for_linear_value0 = float(for_linear_value0)
        self.for_linear_value1 = float(for_linear_value1)
        
    def __call__(self, features):
        '''
        features: input features to the neural network
        '''
        return (self.crossed_term(features) - self.for_linear_term0(features) - self.for_linear_term1(features) + 1) / (self.for_linear_value0*self.for_linear_value1)

class full_likelihood:
    '''
    takes yaml of the form:
        wcs: [wc1, wc2, ...]
        means: [sm, linear, quadratic]
        wc1_quad: path_to_quad_net
        wc1_forlinear: path_to_forlinear_net, value_forlinear_net
        wc1_wc2_comb: path_to_crossed_net
    and returns the likelihood ratio
    '''
    def __init__(self, input_file):
        '''
        input_file: yaml file with the configuration of the likelihood and paths to the neural networks
        '''
        #pull network information from yaml file
        with open(input_file) as f:
            self.configuration = yaml.safe_load( f.read() )
        self.wcs = self.configuration['wcs'].split(",")
        self.quadratic = {}; self.linear={}; value_forlinear={}; net_forlinear={}
        self.no_lin = ['ctu1', 'cqd1', 'cqq13', 'cqu1', 'cqq11', 'ctd1', 'ctq1'] # WCs with no interference with the SM 
        self.sm, self.linr, self.quad = self.configuration['means'].split(",")
        self.sm = float(self.sm); self.linr = float(self.linr); self.quad = float(self.quad)
        #build likelihood ratio for linear and quadratic terms
        for wc in self.wcs:
            self.quadratic[wc] = likelihood_term(torch.load(self.configuration[f'{wc}_quad'], map_location=torch.device('cpu')), self.quad/self.sm)
            if wc not in self.no_lin:
                value_forlinear[wc], net_forlinear[wc] = self.configuration[f'{wc}_forlinear'].split(",")
                self.linear[wc] = linear_term(self.quadratic[wc], 
                                              likelihood_term(torch.load(net_forlinear[wc]), self.linr/self.sm), 
                                              value_forlinear[wc])
        #build likelihood ratio for crossed terms
        if len(self.wcs) > 1:
            self.crossed_net = torch.load(self.configuration[f'{self.wcs[0]}_{self.wcs[1]}_comb'])
            self.crossed = crossed_term(likelihood_term(self.crossed_net),
                                        likelihood_term(torch.load(net_forlinear[self.wcs[0]], map_location=torch.device('cpu'))),
                                        likelihood_term(torch.load(net_forlinear[self.wcs[0]], map_location=torch.device('cpu'))),
                                        value_forlinear[self.wcs[0]], value_forlinear[self.wcs[1]])

    def __call__(self, features, coef_values):
        '''
        features: input features to the neural network
        coef_values: values of the Wilson coefficients to transform the likelihood ratio
        '''
        if set(self.wcs)!=set(coef_values.keys()):
            print(self.wcs, coef_values.keys())
            raise RuntimeError(f"The coefs passed to the likelihood do not align with those used in the likelihood parametrization")
            
        linear_term={}; quadratic_term={}
        likelihood_ratio = torch.ones_like(features[:,0])
        
        for wc in self.wcs:
            quadratic_term[wc] = (self.quadratic[wc](features)*coef_values[wc]*coef_values[wc]).flatten()
            likelihood_ratio   = likelihood_ratio + quadratic_term[wc]
            
            if wc not in self.no_lin:
                linear_term[wc]    = (self.linear[wc](features)*coef_values[wc]).flatten()
                likelihood_ratio   = likelihood_ratio + linear_term[wc]
            
        if len(self.wcs) > 1:
            cross_term       = (self.crossed(features)*coef_values[self.wcs[0]]*coef_values[self.wcs[1]]).flatten()
            likelihood_ratio = likelihood_ratio + cross_term

        return likelihood_ratio
        
