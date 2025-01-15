import os
import gc
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
import scipy.io as sio
from scipy.sparse import coo_matrix, csr_matrix
from math import log
from tqdm import tqdm
import json

# Gaussian decay function for symmetric proximity
def temporal_decay(time_diff, alpha=0.1):
    return np.exp(-alpha * time_diff)
    
#The input items should index from 0
def calculate_tpmi_similarity(item_sequences, alpha):
    # Count the co-occurrences of items
    co_occurrence = defaultdict(float)
    #how many sequences contains item_i
    item_counts = defaultdict(int)
    total_sequences = len(item_sequences)
    for seq in tqdm(item_sequences, desc="Statistical Features Processing..."):
        unique_items = seq    #Our datasets doesn't has duplicate items in the history sequence
        n = len(unique_items)
        for i in range(n):
            item_i = unique_items[i]
            item_counts[item_i] += 1
            for j in range(i+1, n):
                item_j = unique_items[j]
                time_diff = (j - i) - 1
                decay = temporal_decay(time_diff, alpha)
                co_occurrence[(item_i, item_j)] += decay
                co_occurrence[(item_j, item_i)] += decay

    # Convert item sets into a list of unique items
    max_items, min_items = max(item_counts.keys()), min(item_counts.keys())
    print(f'max item id: {max_items}, min item id: {min_items}')
    N = max_items + 1    
    # Initialize lists for sparse matrix creation
    data, row, col = [], [], []
    # PMI calculation
    for (item_i, item_j), cooc_count in tqdm(co_occurrence.items(), desc="PMI Calculating..."):        
        p_ij = cooc_count / total_sequences
        p_i = item_counts[item_i] / total_sequences
        p_j = item_counts[item_j] / total_sequences
        if p_ij > 0:
            # PMI formula, ensure no log(0)
            tpmi = log(p_ij / (p_i * p_j))
            # Store the result in the sparse matrix format
            data.append(tpmi)
            row.append(item_i)
            col.append(item_j)
    return data, row, col, N

def get_user_hist(data_dir, columns_to_read):
    fp = f"train.parquet.gz"
    train = pd.read_parquet(os.path.join(data_dir, fp), columns=columns_to_read)
    #test set ONLY used for get max item id. No data leakage here. All TPMI calculation is based on the training set.
    fp = f"test.parquet.gz"
    test = pd.read_parquet(os.path.join(data_dir, fp), columns=columns_to_read)

    #get user behavior in temporal order
    item_sequences = []
    user_hist_dict = {}
    user_group = train.groupby('uid')
    for u, v in tqdm(user_group, desc='Generating Item Sequence...'):
        curr_hist = None
        curr_rating = None
        max_hist = -1
        for idx, row in v[::-1].iterrows():
            his_num = len(row['history ID'])
            if his_num > max_hist:
                max_hist = his_num
                curr_hist = [iid for iid in row['history ID']] + [row['iid']]
        item_sequences.append(curr_hist)

    N = max(train['iid'].max(), test['iid'].max()) + 1
    print(f"item num: {N}")
    return item_sequences, N

def calculate_tpmi_amazon(args, alpha):
    print('Start Processing...')
    dataset = args.dataset
    data_dir = os.path.join(args.dataset_dir, args.dataset, 'proc_data')
    columns_to_read = ['uid', 'iid', 'history ID']

    item_sequences, N = get_user_hist(data_dir, columns_to_read)
    data, row, col, _ = calculate_tpmi_similarity(item_sequences, alpha=args.alpha)
    print(f'tpmi data num: {len(data)}')
    sparse_matrix = csr_matrix((data, (row, col)), shape=(N, N))
    return sparse_matrix

def calculate_tpmi_ml(args, alpha):
    print('Start Processing...')
    dataset = args.dataset
    data_dir = os.path.join(args.dataset_dir, args.dataset, 'proc_data')    

    #get training set to calculate the TPMI
    fp = f"train.parquet.gz"
    train = pd.read_parquet(os.path.join(data_dir, fp))
    #set item id start from 0
    train['history ID'] = train['history ID'] - 1
    train['Movie ID'] = train['Movie ID'] - 1
    #get total iid num
    id_to_title = json.load(open(os.path.join(data_dir, "id_to_title.json"), "r"))   
    if dataset == 'ml-1m':
        N = train['Movie ID'].max() + 1
    else:
        N = len(id_to_title.keys())
    print(f"item num: {N}")
    
    #get user behavior in temporal order
    item_sequences = []
    user_group = train.groupby('User ID')
    for u, v in tqdm(user_group, desc='Generating Item Sequence'):
        curr_hist = None
        curr_rating = None
        max_hist = -1
        for idx, row in v[::-1].iterrows():
            his_num = len(row['history ID'])
            if his_num > max_hist:
                max_hist = his_num
                curr_hist = row['history ID'].tolist() + [row['Movie ID']] 
        item_sequences.append(curr_hist)
    del train
    gc.collect()
    data, row, col, _ = calculate_tpmi_similarity(item_sequences, alpha=args.alpha)  
    sparse_matrix = csr_matrix((data, (row, col)), shape=(N, N))
    return sparse_matrix