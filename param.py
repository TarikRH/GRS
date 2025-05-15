import os, argparse
# from yaml import safe_load as yaml_load
from json import dumps as json_dumps
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='SDR Arguments')
    parser.add_argument('--desc', type=str, default='')

    # Configuration Arguments
    parser.add_argument('--cuda',   type=str, default='0')
    parser.add_argument('--seed',   type=int, default=2023)

    # Model Arguments
    parser.add_argument('--n_hid',     type=int,   default=64)
    parser.add_argument('--n_layers',  type=int,   default=2)
    parser.add_argument('--s_layers', type=int, default=2)
    parser.add_argument('--weight', type=bool, default=True, help='Add linear weight or not')

    # Train Arguments
    parser.add_argument('--dropout', type=float, default=0)

    # Optimization Arguments
    parser.add_argument('--lr',         type=float, default=25e-4)
    parser.add_argument('--difflr', type=float, default=7e-4)
    parser.add_argument('--reg',        type=float, default=25e-3)
    parser.add_argument('--decay',      type=float, default=0.98)
    parser.add_argument('--decay_step', type=int,   default=2)
    parser.add_argument('--n_epoch',    type=int,   default=15)
    parser.add_argument('--batch_size', type=int,   default=4096)
    parser.add_argument('--patience',   type=int,   default=5)

    # Valid/Test Arguments 
    parser.add_argument('--topk',            type=int, default=20)
    parser.add_argument('--test_batch_size', type=int, default=2048)

    # Data Arguments
    parser.add_argument('--dataset',       type=str, default="epinions")
    parser.add_argument('--num_workers',   type=int, default=0)
    parser.add_argument('--save_name', type=str, default='tem')
    parser.add_argument('--checkpoint',    type=str, default="./Model/epinions/_tem_.pth")
    parser.add_argument('--model_dir',     type=str, default="./Model/epinions/")
   
    # Params for the denoiser
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=int, default=64, help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=True, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=16, help='timestep embedding size')

    # Params for diffusions
    parser.add_argument('--steps', type=int, default=10, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=True, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=5, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

    # Gérer les erreurs de parsing dans les notebooks
    if "__file__" not in globals():
        # On est dans un notebook Jupyter, on ignore les arguments CLI
        args, _ = parser.parse_known_args([])
    else:
        # On est dans un script .py, on utilise les arguments CLI
        args = parser.parse_args()

    return args

import sys

if "ipykernel_launcher" in sys.argv[0]:
    sys.argv = [sys.argv[0]]

args = parse_args()
