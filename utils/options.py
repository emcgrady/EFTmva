import os 
import time 
import yaml 
from argparse import ArgumentParser
def handleOptions():

    parser = ArgumentParser()
    parser.add_argument("--files",type=str, default="/pnfs/psi.ch/cms/trivcat/store/user/sesanche//EFT_mva/ttbar/NanoGen_tt_LO_SMEFTrwgt_v2_postprocess/*.root", help="List of files to process");
    parser.add_argument("--reload",  action='store_true', default=False, help="Force conversion of hdf to pytorch")
    parser.add_argument("--epochs",type=int, default=100, help="Number of epochs to train the net");
    parser.add_argument("--batch-size",type=int, default=64, help="Minibatch size");
    parser.add_argument("--term",  type=str, default=None, help="Train SM against a given term")
    parser.add_argument("--bsm-point",  type=str, default=None, help="Train SM against a given BSM point. Syntax is operator1=value2:operator2=value2:...")
    parser.add_argument("--device", type=str, default='cpu', help="Which device (cpu, gpu index) to use ");
    parser.add_argument("--name", type=str, default="v1", help="Name to store net.");
    parser.add_argument("--learning-rate", type=float, default=0.00000005, help="Optimizer learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--wc-list", type=str, default="sm,ctu1,cqd1,cqq13,ctu8,cqu1,cqq11,cqq83,ctd1,ctd8,ctg,ctq1,cqq81,cqu8,cqd8,ctq8", help="Comma-separated of WC in the sample (by order)")
    parser.add_argument("--features", type=str, default="Lep1_pt,Lep2_pt,Lep1_eta,Lep2_eta,Lep1_phi,Lep2_phi,nJet30,jet1_pt,jet2_pt", help="Comma-separated of WC in the sample (by order)")
    parser.add_argument("--forceRebuild", action="store_true", default=False, help="Force reproduction of torch tensors from rootfiles")
    parser.add_argument("--configuration-file", type=str, default=None, help="Load parameters from toml configuration file. The configuration file will be overriden by other command line options. The --name argument will always be taken from the command line option and the default")
    args = parser.parse_args()
    
    directory_name = time.strftime('%Y%m%d-%H%M%S') + "_" + args.name
    os.system(f"mkdir {directory_name}")


    # A bit of juggling with the configuration so we store it in yml files
    if args.configuration_file:
        with open(args.configuration_file) as f:
            config = yaml.safe_load(f.read())
    else:
        config = {}
    config = {**config, **vars(args)}
    
    with open(f"{directory_name}/config.yml","w") as f:
        f.write( yaml.dump(config)) 

    for op, val in config.items():
        setattr( args, op, val)
        
    return args, directory_name

def rocOptions():
    parser = ArgumentParser()
    parser.add_argument("--signal", type=str, default=None, 
                        help="directory of signal sample")
    parser.add_argument("--likelihood", type=str, default=None, 
                        help="yaml file containing information needed for building likeliood")
    parser.add_argument("--name", type=str, default="v1", 
                        help="Name of directory to store plots")
    parser.add_argument("--test-point",  type=str, default=None, 
                        help="Train SM against a given BSM point. Syntax is operator1=value2:operator2=value2:...")
    args = parser.parse_args()
    
    with open(f'{args.signal}/config.yml') as f:
        config = yaml.safe_load(f.read())
    del config['name']
    for key in config.keys():
        parser.add_argument(f'--{key}' , default=config[key])
    args = parser.parse_args()
    
    directory_name = time.strftime('%Y%m%d-%H%M%S') + "_" + args.name
    os.system(f"mkdir {directory_name}")
    
    if args.configuration_file:
        with open(args.configuration_file) as f:
            config = yaml.safe_load(f.read())
    else:
        config = {}
    config = {**config, **vars(args)}
    
    with open(f"{directory_name}/config.yml","w") as f:
        f.write( yaml.dump(config)) 

    for op, val in config.items():
        setattr( args, op, val)
        
    return args, directory_name
