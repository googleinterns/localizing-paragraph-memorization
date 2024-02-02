# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os, toolz, functools, random, torch, itertools, glob

import torch
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = "/home/jupyter/paraMem/"

## DATA LOADING____________________________________________________________________________

def train_test_batching(mem_prompts:torch.tensor, non_mem_prompts:torch.tensor, mem_batch:int=1, non_mem_batch:int=5, test_frac:float=0.0, shuffle:bool=True, add_bos:int=None, set_twice:str=None):    
    
    if add_bos is not None and isinstance(add_bos, int):
        print(f"prepending bos token id {add_bos} to get shape {mem_prompts.shape[-1] + 1}")
        mem_prompts = add_bos_token(mem_prompts, bos_tok_id = add_bos)
        non_mem_prompts = add_bos_token(non_mem_prompts, bos_tok_id = add_bos)
    
    ## for baseline experiments, compare non-mem with non-mem sets
    if set_twice is not None:
        if set_twice=="c":
            non_mem_prompts = mem_prompts.clone()
        elif set_twice=="k":
            mem_prompts = non_mem_prompts.clone()
    
    ## split in train and test set
    n_mem, n_non_mem = mem_prompts.shape[0], non_mem_prompts.shape[0]
    n_test_mem, n_test_non_mem = int(test_frac*n_mem), int(test_frac*n_non_mem)
    train_mem, test_mem = torch.utils.data.random_split(mem_prompts, [n_mem-n_test_mem, n_test_mem])
    train_non_mem, test_non_mem = torch.utils.data.random_split(non_mem_prompts, [n_non_mem-n_test_non_mem, n_test_non_mem])

    train_mem_dl = torch.utils.data.DataLoader(train_mem, batch_size=mem_batch, shuffle=shuffle)
    train_non_mem_dl = torch.utils.data.DataLoader(train_non_mem, batch_size=non_mem_batch, shuffle=shuffle)
    
    test_mem_dl, test_non_mem_dl = iter([]), iter([])
    if test_frac > 0: ## empty iterable if test_frac == 0.0
        test_mem_dl = torch.utils.data.DataLoader(test_mem, batch_size=mem_batch, shuffle=shuffle)
        test_non_mem_dl = torch.utils.data.DataLoader(test_non_mem, batch_size=non_mem_batch, shuffle=shuffle)

    train_dl = torch.utils.data.DataLoader(list(zip(itertools.cycle(train_mem_dl), train_non_mem_dl)), batch_size=1, shuffle=False)
    test_dl = torch.utils.data.DataLoader(list(zip(itertools.cycle(test_mem_dl), test_non_mem_dl)), batch_size=1, shuffle=False)
    return train_dl, test_dl


## DATA PREPROCESSING______________________________________________________________


def load_perturbed_mem(file_path:str="acc/gpt-neo-125M"):
    mem_toks_NI = torch.load(f"/home/jupyter/paraMem/data/pile_splits/{file_path}/mem_toks.pt")
    mem_perturbed_toks_NI = torch.load(f"/home/jupyter/paraMem/data/pile_splits/{file_path}/perturbed_mem_toks.pt")
    return mem_toks_NI, mem_perturbed_toks_NI

def load_pile_splits(path:str="acc/gpt-neo-125M", file_names:list=["mem.pt", "non_mem.pt"], as_torch:bool=False):
    
    prompts, counts = load_pile_seqs(seq_length=100)
    if as_torch:
        prompts, counts = torch.LongTensor(prompts), torch.LongTensor(counts)

    folder_path = Path(ROOT + "/data/pile_splits") / str(path)
    if file_names is None:
        file_names = glob.glob(f"{folder_path}/*")
        file_names = list(map(lambda file_name: file_name.split("/")[-1], file_names))
        file_names.sort()

    print(f"from {folder_path} loading {file_names[:]}...")
    prompts_counts_list = []
    for file_name in file_names:
        pile_idcs = torch.load(folder_path / file_name)
        prompts_counts = (prompts[pile_idcs], counts[pile_idcs])
        prompts_counts_list.append(prompts_counts)
    return prompts_counts_list

def load_pile_seqs(seq_length=100):
    data_folder = ROOT + "paraMem/data/lm_mem"
    data_files = os.listdir(data_folder)
    prompts, counts = None, None    
    for f in data_files:
        f_seq_length = f.split("_")[1].split(".")[0]
        if f_seq_length == str(seq_length):
            f_type = f.split("_")[0]
            if "counts" in f_type: ## number of prefixes (and their counts)
                counts = np.load(Path(data_folder) / f)
            elif "prompts" in f_type: ## number of prefixes, sequence length
                #prompts = np.memmap(Path(data_folder) / f)
                prompts = np.load(Path(data_folder) / f)
    return prompts, counts


def preprocess_pile_seqs(seq_length=100, n_seqs=None, uni_tok_frac=0.0, filter_toks:list=None, count_ranges=None, k_uniform=None):

    prompts, counts = load_pile_seqs(seq_length=seq_length)
    data_indices = np.array(range(0,len(prompts)))
    metadata={"seq_length":seq_length, "uni_tok_frac":uni_tok_frac, "filter_toks": filter_toks.to("cpu"), "count_ranges": count_ranges, "k_uniform":k_uniform}

    ## (1) pre-filtering________________________________________
    
    non_zero_idcs = np.argwhere(counts>0).squeeze() ## remove sequences with 0 count
    min_unique_toks = int(uni_tok_frac * prompts.shape[-1])
    n_unique_toks = np.apply_along_axis(toolz.compose(len, np.unique), 1, prompts)
    non_unique_idcs = np.argwhere(n_unique_toks>min_unique_toks).squeeze() ## remove sequences where all tokens are the same

    keep_idcs = np.array(list(functools.reduce(np.intersect1d, (non_unique_idcs, non_zero_idcs))))
    prompts, counts, data_indices = prompts[keep_idcs], counts[keep_idcs], data_indices[keep_idcs]
    
    ## (2) token filters___________________________________________


    if filter_toks is not None:
        eos_token_id = 50256
        print(f"filtering tokens and eos: {eos_token_id}")
        filter_toks_mask = torch.where(filter_toks==eos_token_id, 0, 1) ## remove EOS tokens
        filter_sets = [set(torch.masked_select(filter_toks[i,:], filter_toks_mask.bool()[i,:]).tolist()) for i in range(filter_toks.shape[0])]
        keep_idcs = [i for i, prompt in enumerate(prompts) if not any(filter_seq.issubset(set(prompt.tolist())) for filter_seq in filter_sets)]
        prompts, counts, data_indices = prompts[keep_idcs], counts[keep_idcs], data_indices[keep_idcs]
        
    ## (3) sampling___________________________________________

    if k_uniform is not None:
        print(f"k_uniform sampling {k_uniform}")
        #unique_values, counts = np.unique(a, return_counts=True) # Count occurrences of each unique value in the array
        #valid_indices = np.where(np.isin(a, unique_values[counts >= k]))[0] # Filter indices of values that occur at least k times
        unique_values, value_counts = np.unique(counts, return_counts=True)
        valid_values = unique_values[value_counts >= k_uniform]
        mask = np.zeros_like(counts, dtype=bool)
        for val in valid_values:
            indices = np.where(counts == val)[0]
            mask[indices[:k_uniform]] = True
            keep_idcs = np.nonzero(mask)
        prompts, counts, data_indices = prompts[keep_idcs], counts[keep_idcs], data_indices[keep_idcs]
        
    if count_ranges is not None:
        print(f"count_ranges {count_ranges}")
        keep_idcs = list()
        for count_range in count_ranges:
            keep_idcs += np.where((counts >= count_range[0]) & (counts <= count_range[1]))[0].tolist()
        keep_idcs = np.array(list(set(keep_idcs)))
        prompts, counts, data_indices = prompts[keep_idcs], counts[keep_idcs], data_indices[keep_idcs]
        
    ## (4) shuffling and select_________________________________________
    
    #np.random.seed(0)
    shuffle_idcs = np.arange(len(counts))
    np.random.shuffle(shuffle_idcs)
    prompts, counts, data_indices = prompts[shuffle_idcs], counts[shuffle_idcs], data_indices[shuffle_idcs]  
    
    data_indices = torch.LongTensor(data_indices[:n_seqs])
    data_indices.meta = metadata
    return torch.tensor(prompts[:n_seqs]), torch.tensor(counts[:n_seqs]), data_indices


def store_tensor(data_idcs:torch.LongTensor, file_name:str, idcs_N:torch.LongTensor=None, add_meta:dict=None):
    data_path = Path("/home/jupyter/paraMem/data/") / str(file_name)
    if idcs_N is not None:
        data_idcs = idcs_N[data_idcs]
    if add_meta is not None:
        data_idcs.meta = idcs_N.meta
        for k,v in add_meta.items():
            data_idcs.meta[k] = v
    torch.save(data_idcs, data_path)
  
def load_tensor(file_name:str):
    data_path = Path("/home/jupyter/paraMem/data/") / str(file_name)
    data_tensor = torch.load(data_path)
    return data_tensor


def add_bos_token(toks_NI:torch.tensor, bos_tok_id:int=50256):
    """
    prepending the <bos> token to all sequences in the given tensor if it is not there already
    """
    if toks_NI[0,0] != bos_tok_id: 
        bos_N = torch.tensor([bos_tok_id]).repeat(toks_NI.shape[0]).unsqueeze(1)
        toks_NI = torch.cat((bos_N, toks_NI), dim=-1)
    return toks_NI


def expand_filter_list(filter_str:list):
    """
    expanding a string list with string variants such as capitalization
    """
    filter_str_expanded = [(term, term.upper(), term.capitalize()) for term in filter_str]
    filter_str_expanded = list(itertools.chain.from_iterable(filter_str_expanded))
    filter_str_expanded += [" " + term for term in filter_str_expanded]
    return filter_str_expanded


if __name__ == '__main__':
    prompts, counts = load_pile_seqs(seq_length=100, n_seqs=100, unique_frac=0.5, k_uniform=1)
    print(f"{prompts}, {counts}")