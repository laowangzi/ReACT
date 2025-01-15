import os, argparse
import numpy as np
import pickle
import h5py
from scipy.sparse import csr_matrix
import scipy.io as sio
from utils.calculate_tpmi_sim import calculate_tpmi_amazon, calculate_tpmi_ml

def tpmi_sim(args):    
    if args.dataset in ['ml-1m', 'ml-100k']:
        command = f'calculate_tpmi_ml(args, {args.alpha})'
    elif args.dataset in ['amazon-book-2018']:
        command = f'calculate_tpmi_amazon(args, {args.alpha})'
    csr = eval(command)
    # Save the CSR matrix components to an HDF5 file
    fp =os.path.join(args.save_dir, '_'.join([args.dataset, f'alpha{args.alpha}', 'csr_matrix'])+".h5")
    with h5py.File(fp, 'w') as f:
        f.create_dataset('data', data=csr.data)
        f.create_dataset('indices', data=csr.indices)
        f.create_dataset('indptr', data=csr.indptr)
        f.create_dataset('shape', data=csr.shape)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--dataset_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="./tpmi_results")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    tpmi_sim(args)

