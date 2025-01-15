import os, argparse
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from math import log
from tqdm import tqdm
import json
import pickle
import h5py
import scipy.sparse as sp
from transformers import AutoTokenizer
import heapq
import random

#min length MIN_N = filtered minimize interaction num
sparse_matrix, P, MAX_N, K, padding_value, MIN_N, id_to_title = None, None, None, None, None, None, None

def get_template(input_dict):
    nominative = "she" if input_dict["Gender"]=="female" else "he"
    objective = "Her" if input_dict["Gender"]=="female" else "His"
    template = \
{        "template": 
    f"The user is a {input_dict['Gender']}. "
    f"{objective} job is {input_dict['Job']}. {objective} age is {input_dict['Age']}.\n"
    f"{nominative.capitalize()} watched the following movies in order in the past, and rated them:\n"
    f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['history ID'][::])))}\n"
    f"Based on the movies {nominative} has watched, deduce if {nominative} will like the movie ***{input_dict['Movie ID']}***.\n"
    f"Note that more stars the user rated the movie, the more the user liked the movie.\n"
    f"You should ONLY tell me yes or no.",
}
    return template["template"]

def generate_input(row):
    global id_to_title
    hist = row['history ID'].tolist()
    input_dict = {}
    input_dict['Gender'], input_dict['Age'], input_dict['Job'] = row['Gender'], row['Age'], row['Job']
    input_dict["history ID"], input_dict["history rating"] = row['retrieval_items'],  row['retrieval_rating']
    #re-order retrieval items into temporal order
    hist_seq_dict = {h: i for i, h in enumerate(hist)}
    zipped_list = sorted(zip(input_dict["history ID"], input_dict["history rating"]), key=lambda x: hist_seq_dict[x[0]])
    input_dict["history ID"], input_dict["history rating"] = map(list, zip(*zipped_list))
    
    input_dict["Movie ID"] = id_to_title[str(row["Movie ID"])]
    input_dict["history ID"] = list(map(lambda index: id_to_title[str(index)], input_dict["history ID"]))

    for i, (name, star) in enumerate(zip(input_dict["history ID"], input_dict["history rating"])):
        suffix = " stars)" if star > 1 else " star)"
        input_dict["history ID"][i] = f"{name} ({star}" + suffix
    return get_template(input_dict)

def top_k_tuples(lst, k):
    # Use a min-heap of size K to store the top K tuples by 'b'
    heap = []    
    for item in lst:
        # Push current item into heap (using 'b' as the key for comparison)
        heapq.heappush(heap, (item[1], item))        
        # If heap size exceeds K, pop the smallest element (maintain only top K)
        if len(heap) > k:
            heapq.heappop(heap)    
    # Extract elements from the heap and sort them in descending order of 'b'
    result = [heapq.heappop(heap)[1] for _ in range(len(heap))]
    return result

def dynamic_K(arr, p, history, has_values):
    global sparse_matrix, P, MAX_N, padding_value, MIN_N
    flag = 0
    if len(has_values) > 1:
        mu = np.mean(has_values[1])
        sigma = np.std(has_values[1])
        threshold = mu + p * sigma
        greater_than_mean = [(i, arr[idx]) for idx, i in enumerate(history) if arr[idx] > threshold]
        if len(greater_than_mean) < MIN_N:
            flag = 1
        K = min(len(greater_than_mean), MAX_N)
        K = max(K, MIN_N)
        all_hist = greater_than_mean
    else:
        all_hist = [(i, arr[idx]) for idx, i in enumerate(history)]
        K = MIN_N
        
    if flag == 0:
        sorted_items = top_k_tuples(all_hist, K)
    else:
        sorted_items = greater_than_mean
        tpmi_items = [tp[0] for tp in sorted_items]
        #recent padding; history must in temporal order
        for it in history[::-1]:
            if it not in tpmi_items:
                sorted_items.append((it, 0))
            if len(sorted_items) == K:
                break
    return sorted_items

def get_dynamic_retrieval(input_dict):
    global sparse_matrix, P, MAX_N, padding_value, MIN_N
    cur_r = []
    cur_r_rating = []
    cur_id = int(input_dict["Movie ID"])
    his_rating = {h:r for h, r in zip(input_dict["history ID"], input_dict["history rating"])}
    #matrix index start from 0
    candidate = cur_id - 1
    history = [(int(i)-1) for i in input_dict["history ID"]]

    candidate_row = sparse_matrix.getrow(candidate)
    # Initialize the list for storing the values for the history indices
    values = []
    #0 for hist idx; 1 for tpmi value
    has_values = [[], []]
    # For each index in history, check if the value exists in the sparse matrix
    for idx, i in enumerate(history):
        # Check if the index i is present in the non-zero indices of the candidate row
        if i in candidate_row.indices:
            # If the value exists in the sparse matrix, use it
            val = candidate_row[0, i]
            has_values[0].append(i)
            has_values[1].append(val)
        else:
            # If the value doesn't exist (implicitly zero), use the minimum value
            #Give padding value a time punishment
            val = padding_value - ((len(history)-idx)*1)
        values.append(val)
    values = np.array(values)
    #Retrieve relevant items from history based on the candidate and TPMI value.
    sorted_items = dynamic_K(values, P, history, has_values)
    for item in sorted_items:
        real_id = item[0] + 1
        cur_r.append(real_id)
        cur_r_rating.append(his_rating[real_id])    
    return np.array(cur_r), np.array(cur_r_rating)

def get_recent_retrieval(input_dict):
    global K
    hist_rating_dict = input_dict["history ID"]
    hist_ratings = input_dict["history rating"]
    cur_r = hist_rating_dict[-K:]
    cur_r_rating = hist_ratings[-K:]
    return np.array(cur_r), np.array(cur_r_rating)

def get_random_retrieval(input_dict):
    global K
    hist_rating_dict = input_dict["history ID"]
    hist_ratings = input_dict["history rating"]
    numK = min(K, len(hist_rating_dict))
    random_index = np.random.choice(list(range(len(hist_rating_dict))), size=numK, replace=False)
    cur_r = hist_rating_dict[random_index]
    cur_r_rating = hist_ratings[random_index]
    return np.array(cur_r), np.array(cur_r_rating)

def construct_input(row):
    data_dict = {}
    labels = row['labels']
    data_dict['input'] = row['msg']
    data_dict['output'] = "Yes." if int(labels) == 1 else "No."
    return data_dict

def retrieve_behavior(args):
    global sparse_matrix, P, MAX_N, K, padding_value, MIN_N, id_to_title
    data_dir = os.path.join(args.dataset_dir, args.dataset, 'proc_data')
    print(f'Reading Data...')
    title_dir=args.id2title
    id_to_title = json.load(open(os.path.join(data_dir, title_dir), "r"))    
    fp = f"{args.target_set}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp))
    
    print(f"Processing...")
    if args.temp_type == 'dynamic':
        # Load the CSR matrix components from the HDF5 file
        fp =os.path.join('./tpmi_results', '_'.join([args.dataset, f'alpha{args.alpha}', 'csr_matrix'])+".h5")
        with h5py.File(fp, 'r') as f:
            data = f['data'][:]
            indices = f['indices'][:]
            indptr = f['indptr'][:]
            shape = tuple(f['shape'][:])
        sparse_matrix = sp.csr_matrix((data, indices, indptr), shape=shape)
        padding_value = sparse_matrix.data.min() - 0.01
        print(f'padding value: {padding_value}')        
        P, MAX_N, MIN_N = args.p, args.max_num, args.min_num
        df['retrieval_items'],  df['retrieval_rating']= zip(*df.apply(get_dynamic_retrieval, axis=1))
    elif args.temp_type in ['recent', 'random', 'range']:
        K = args.K
        if args.temp_type == 'recent':
            df['retrieval_items'],  df['retrieval_rating']= zip(*df.apply(get_recent_retrieval, axis=1))
        elif args.temp_type == 'random':
            df['retrieval_items'],  df['retrieval_rating']= zip(*df.apply(get_random_retrieval, axis=1))
        elif args.temp_type == 'range':
            MAX_N, MIN_N = args.max_num, args.min_num
            df['retrieval_items'],  df['retrieval_rating']= zip(*df.apply(get_range_retrieval, axis=1))
    else:
        print(f'No Implement {args.temp_type}')
        exit()

    df['dretrieval_len'] = df['retrieval_items'].apply(lambda x: len(x))
    print(f"Mean User Sequence Length: ", round(df['dretrieval_len'].mean(), 2))
    print(f"Median User Sequence Length: ", df['dretrieval_len'].median())
    print(f"Max User Sequence Length: ", df['dretrieval_len'].max())
    print(f"Min User Sequence Length: ", df['dretrieval_len'].min())
    print(f"3/4 Quantile User Sequence Length: ", df['dretrieval_len'].quantile(0.75))
    print(f"Generating Prompt and Saving...")
    df['msg'] = df.apply(generate_input, axis=1)
    #sort the input to get a higer inference efficiency
    if args.target_set == 'test' and args.temp_type == 'dynamic':
        # df['msg_length'] = df['msg'].apply(lambda x: len(x))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, add_eos_token=True)
        df['msg_length'] = df['msg'].apply(lambda x: len(tokenizer.encode(x)))
        df_sort = df.sort_values(by=['msg_length'], ascending=True)
        df_sort = df_sort.reset_index(drop=True)
        df_sort.to_parquet(os.path.join(data_dir, f"test_{args.temp_type}_MaxNum{args.max_num}_MinNum{args.min_num}_p{args.p}_tpmi_alpha{args.alpha}.parquet.gz"), compression="gzip")
        data_list = list(df_sort.apply(construct_input, axis=1))
    else:
        data_list = list(df.apply(construct_input, axis=1))
    assert len(data_list) == len(df)
    if args.temp_type == 'dynamic':
        fp = os.path.join(data_dir, f'data/{args.target_set}', f"{args.target_set}_{args.temp_type}_MaxNum{args.max_num}_MinNum{args.min_num}_p{args.p}_tpmi_alpha{args.alpha}.json")
    else:
        fp = os.path.join(data_dir, f'data/{args.target_set}', f"{args.target_set}_K{args.K}_{args.temp_type}.json")
    json.dump(data_list, open(fp, "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--id2title", type=str, default="id_to_title.json")
    parser.add_argument("--dataset_dir", type=str, default="../data")
    parser.add_argument("--tokenizer_dir", type=str, default="LLM_path")
    parser.add_argument("--temp_type", type=str, default="dynamic", help='dynamic/recent/random')
    parser.add_argument("--target_set", type=str, default="test", help="train/valid/test")
    #TPMI params
    parser.add_argument("--tpmi_dir", type=str, default="./tpmi_results")
    parser.add_argument("--alpha", type=float, default=0.05)
    # dynamic parames
    parser.add_argument("--max_num", type=int, default=60)
    parser.add_argument("--min_num", type=int, default=5)
    parser.add_argument("--p", type=float, default=1.5)
    #recent/random params
    parser.add_argument("--K", type=int, default=15, help='topK for recent/random setting')
    args = parser.parse_args()
    print(args)
    retrieve_behavior(args)

