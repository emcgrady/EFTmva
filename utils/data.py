import os
import glob
import torch
import torch.utils.data as data
import uproot
import numpy as np 
import tqdm

class eftDataLoader( data.Dataset ):
    def __init__(self, args):

        self.files = glob.glob(args.files)
        self.wc_list = args.wc_list.split(",")
        self.dtype       = np.float32 # could be configurable
        
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
            self.bsm_name = "bsm_weight" +  self.term
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
                sm_weight = eft_coefficients[:,0].to_numpy()
                outputs['sm_weight']  = np.append( outputs['sm_weight'], sm_weight )

            if redoBSM:
                if self.term is not None:
                    bsm_weight = eft_coefficients[:,self.coef_map[tuple(self.term.split("_"))]].to_numpy()
                else:
                    coef_values = self.bsm_point.split(':')
                    bsm_weight = eft_coefficients[:,0].to_numpy()
                    for i1, coef_value in enumerate(coef_values):
                        coef,value = coef_value.split("="); value = float(value)
                        bsm_weight += eft_coefficients[:,self.coef_map[(coef,'sm')]].to_numpy()*value       # linear term
                        bsm_weight += eft_coefficients[:,self.coef_map[(coef,coef)]].to_numpy()*value*value # quadratic term
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
            torch.save( t, f'/scratch/{output}.p') # can certainly be improved
            os.system(f'xrdcp /scratch/{output}.p root://t3dcachedb.psi.ch:1094//{self.out_path}/')
            os.system(f'rm /scratch/{output}.p')

    def load_tensors(self):
        self.sm_weight  = torch.load( f'{self.out_path}/sm_weight.p').to(device = self.device)
        self.bsm_weight = torch.load( f'{self.out_path}/{self.bsm_name}.p').to(device = self.device)
        self.features   = torch.load( f'{self.out_path}/features.p').to(device = self.device)


if __name__=="__main__":
    
    data = eftDataLoader("/pnfs/psi.ch/cms/trivcat/store/user/sesanche//EFT_mva/ttbar/NanoGen_tt_LO_SMEFTrwgt_v2_postprocess/*.root", term=("ctq8","sm"), bsm_point=None, device='cpu', forceRebuild=True, wc_list=["sm","ctu1","cqd1","cqq13","ctu8","cqu1","cqq11","cqq83","ctd1","ctd8","ctg","ctq1","cqq81","cqu8","cqd8","ctq8"], feature_list=['Lep1_pt','Lep1_eta']  )
    print(len(data))
    print(data[2])
