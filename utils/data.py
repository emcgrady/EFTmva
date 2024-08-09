from torch.utils.data import DataLoader

import torch.utils.data as data
import numpy as np 

import uproot
import torch
import glob
import tqdm
import os


class eftDataLoader( data.Dataset ):
    def __init__(self, args):

        self.files   = glob.glob(args.files)
        self.wc_list = args.wc_list.split(",")
        self.dtype   = np.float64
        
        if args.term is None and args.bsm_point is None:
            raise RuntimeError("You need to decide whether you get the weights associated to a given term or to a given bsm point")

        if not args.term is None and not args.bsm_point is None:
            raise RuntimeError("You need to decide whether you either get the weights associated to a term")
            
        self.term =  args.term
        self.bsm_point = args.bsm_point
        self.feature_list  = args.features.split(',')
        self.out_path = "/".join(self.files[0].split("/")[:-2])
        self.forceRebuild = args.forceRebuild
        self.device = args.device

        self.buildMapping()
        self.build_tensors()
        self.load_tensors()

    def __len__( self ):
        return self.sm_weight.shape[0]

    def __getitem__(self, idx):
        return self.sm_weight[idx], self.bsm_weight[idx], self.features[idx,:]

    def buildMapping( self ):

        self.coef_map={}
        index=0
        for i in range(len(self.wc_list)):
            for j in range(i+1):
                self.coef_map[(self.wc_list[i],self.wc_list[j])]=index
                index+=1

                
    def build_tensors( self ):

        if self.term is not None:
            self.bsm_name = "bsm_weight_" +  self.term
        else:
            self.bsm_name = "bsm_weight_" + self.bsm_point.replace("=","_").replace(":","_")

        if self.forceRebuild:
            os.system(f'rm -f {self.out_path}/*.p')

        redoSM  = not os.path.isfile(f'{self.out_path}/sm_weight.p')
        redoBSM = not os.path.isfile(f'{self.out_path}/{self.bsm_name}.p')
        redoFeatures = not os.path.isfile(f'{self.out_path}/features.p')

        outputs={}
        if redoFeatures:
            print("Will redo tensor with input features")
            outputs['features'        ] = np.empty( shape=(0, len(self.feature_list)), dtype=self.dtype)
        if redoSM:
            print("Will redo tensor with SM weight")
            outputs[ 'sm_weight'      ] = np.empty( shape=(0), dtype=self.dtype)
        if redoBSM:
            print("Will redo tensor with BSM weight")
            outputs[ self.bsm_name    ] = np.empty( shape=(0), dtype=self.dtype)

        if not (redoBSM or redoSM or redoFeatures):
            return 

        print("Loading files, this may take a while")
        for fil in tqdm.tqdm(self.files):
            tf = uproot.open( fil )
            events = tf["Events"]

            # First we read the EFT stuff 
            if redoSM or redoBSM:
                eft_coefficients=events["EFTfitCoefficients"].array()

            if redoSM:
                #filter out small quadratic terms
                sm_weight = eft_coefficients[:,0].to_numpy()
                outputs['sm_weight']  = np.append( outputs['sm_weight'], sm_weight )

            if redoBSM:
                if self.term is not None:
                    quad_term = eft_coefficients[:,self.coef_map[tuple(self.term.split("_"))]].to_numpy()
                    quad_term[np.abs(quad_term/sm_weight) < 1e-4] = 0
                    
                    bsm_weight = quad_term
                else:
                    coef_values = self.bsm_point.split(':')
                    bsm_weight  = eft_coefficients[:,0].to_numpy()
                    sm_weight   = eft_coefficients[:,0].to_numpy()
                    for i1, coef_value in enumerate(coef_values):
                        coef,value = coef_value.split("="); value = float(value)

                        #filter out small linear terms

                        linr_term = eft_coefficients[:,self.coef_map[(coef,'sm')]].to_numpy()
                        linr_term[np.abs(linr_term/sm_weight) < 1e-4] = 0
                        
                        bsm_weight += linr_term*value       # linear term

                        #filter out small quadratic terms
                        quad_term = eft_coefficients[:,self.coef_map[(coef,coef)]].to_numpy()
                        quad_term[np.abs(quad_term/sm_weight) < 1e-4] = 0
                        
                        bsm_weight += quad_term*value*value # quadratic term
                        
                        for i2, coef_value2 in enumerate(coef_values):
                            if i2 >= i1: continue
                            coef2,value2 = coef_value2.split("="); value2=float(value2)
                            idx = self.coef_map[(coef,coef2)] if (coef,coef2) in self.coef_map else self.coef_map[(coef2, coef)]
                            bsm_weight += eft_coefficients[:,idx].to_numpy()*value*value2 # crossed terms

                
                outputs[self.bsm_name] = np.append( outputs[self.bsm_name], bsm_weight )

            if redoFeatures:
                features =  events.arrays(self.feature_list, library='pandas').to_numpy()
                outputs['features'] = np.append( outputs['features'], features, axis=0)
            #break # for development

        # writing tensors to file
        for output in outputs:
            t = torch.from_numpy( outputs[output] )
            torch.save( t, f'{self.out_path}/{output}.p')

    def load_tensors(self):
        self.sm_weight  = torch.load( f'{self.out_path}/sm_weight.p').to(device = self.device)
        self.bsm_weight = torch.load( f'{self.out_path}/{self.bsm_name}.p').to(device = self.device)
        self.features   = torch.load( f'{self.out_path}/features.p').to(device = self.device)
