import os 
import time 
import yaml 
from argparse import ArgumentParser

def commonOptions():

    parser = ArgumentParser()
    parser.add_argument("--files",type=str, default="/pnfs/psi.ch/cms/trivcat/store/user/sesanche//EFT_mva/ttbar/NanoGen_tt_LO_SMEFTrwgt_v2_postprocess/*.root", help="List of files to process");
    parser.add_argument("--reload",  action='store_true', default=False, help="Force conversion of hdf to pytorch")
    parser.add_argument("--device", type=str, default='cpu', help="Which device (cpu, gpu index) to use ");
    parser.add_argument("--name", type=str, default="v1", help="Name to store net.");
    parser.add_argument("--wc-list", type=str, default="sm,ctu1,cqd1,cqq13,ctu8,cqu1,cqq11,cqq83,ctd1,ctd8,ctg,ctq1,cqq81,cqu8,cqd8,ctq8", help="Comma-separated of WC in the sample (by order)")
    parser.add_argument("--features", type=str, default="Lep1_pt,Lep2_pt,Lep1_eta,Lep2_eta,Lep1_phi,Lep2_phi,nJet30,jet1_pt,jet2_pt", help="Comma-separated of WC in the sample (by order)")
    parser.add_argument("--forceRebuild", action="store_true", default=False, help="Force reproduction of torch tensors from rootfiles")
    parser.add_argument("--configuration-file", type=str, default=None, help="Load parameters from toml configuration file. The configuration file will be overriden by other command line options. The --name argument will always be taken from the command line option and the default")
    return parser


def parse(parser):

    args = parser.parse_args()

    # A bit of juggling with the configuration so we store it in yml files
    if args.configuration_file:
        with open(args.configuration_file) as f:
            config = yaml.safe_load(f.read())
    else:
        config = {}
    config = {**config, **vars(args)}
    
    os.system(f"mkdir -p {args.name}")


    with open(f"{args.name}/config.yml","w") as f:
        f.write( yaml.dump(config)) 

    for op, val in config.items():
        setattr( args, op, val)

    return args

def handleOptions():

    parser = commonOptions()
    parser.add_argument("--epochs",type=int, default=100, help="Number of epochs to train the net");
    parser.add_argument("--bsm-point",  type=str, default=None, help="Train SM against a given BSM point. Syntax is operator1=value2:operator2=value2:...")
    parser.add_argument("--batch-size",type=int, default=64, help="Minibatch size");
    parser.add_argument("--term",  type=str, default=None, help="Train SM against a given term")
    parser.add_argument("--learning-rate", type=float, default=0.00000005, help="Optimizer learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--profile", type=bool, default=False, help="Set to True to profile the network")

    return parse(parser)

def rocOptions():
    parser = commonOptions()
    parser.add_argument("--likelihood", type=str, default=None, 
                        help="yaml file containing information needed for building likeliood")
    parser.add_argument("--bsm-point",  type=str, default=None, 
                        help="BSM point that is going to be signal in the roc curve")
    parser.add_argument("--term",  type=str, default=None,
                        help="Term that is going to be signal in the roc curve")
    parser.add_argument("--dedicated",  type=str, default=None, 
                        help="Path to a network trained against an specific BSM point ")
    
    
    return parse(parser)
