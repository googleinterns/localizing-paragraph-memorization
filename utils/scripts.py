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

## DATALOADERS________________________

import sys, os, toolz, functools, random, torch, itertools

import torch
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = "/home/jupyter"

## DATA LOADING____________________________________________________________________________

def train_test_batching(mem_prompts:torch.tensor, non_mem_prompts:torch.tensor, mem_batch:int=1, non_mem_batch:int=5, test_frac:float=0.0, shuffle:bool=True, add_bos:int=None, set_twice:str=None):    
    
    if add_bos is not None and isinstance(add_bos, int):
        print(f"prepending bos token id {add_bos} to get shape {mem_prompts.shape[-1] + 1}")
        mem_prompts = add_bos_token(mem_prompts, bos_tok_id = add_bos)
        non_mem_prompts = add_bos_token(non_mem_prompts, bos_tok_id = add_bos)
    
    ## for baseline experiments, compare non-mem with non-mem sets
    if set_twice is not None:
        if set_twice=="c":
            non_mem_prompts = mem_prompts
        elif set_twice=="k":
            mem_prompts = non_mem_prompts
    
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

def load_pile_splits(folder:str, as_torch:bool=False):
    folder_path = Path(ROOT + "/paraMem/data/pile_splits") / str(folder)
    mem_idcs, non_mem_idcs = torch.load(folder_path / "mem.pt"), torch.load(folder_path / "non_mem.pt")
    seq_length = int(mem_idcs.meta["seq_length"]) ## read off seq length
    prompts, counts = load_pile_seqs(seq_length=seq_length)
    if as_torch:
        prompts, counts = torch.LongTensor(prompts), torch.LongTensor(counts)
    mem_prompts_counts = (prompts[mem_idcs], counts[mem_idcs])
    non_mem_prompts_counts = (prompts[non_mem_idcs], counts[non_mem_idcs])
    return mem_prompts_counts, non_mem_prompts_counts

def load_pile_seqs(seq_length=100):
    data_folder = ROOT + "/paraMem/data/lm_mem"
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
    
    
    
## EVALUATION_______________________________________________________________________________

import torch, tqdm

import sys
sys.path.append('/home/jupyter/')
from paraMem.utils import modelHandlers


def evaluate_nll_greedy(model, toks_NI:torch.LongTensor, n_greedy:int=50, batch_size:int=5, tqdm_disable:bool=True):
    """
    batching the token sequence to run them through the model and evaluate it on NLL and greedy decoding
    """
    if len(toks_NI.shape)>=3: ## remove outer batch dimension
        toks_NI = toks_NI.squeeze(0)

    with torch.no_grad():
        all_nll_NI, minK_nll_NI = torch.zeros(toks_NI.shape[0], toks_NI.shape[1]-1), torch.empty(toks_NI.shape[0], int(0.2*toks_NI.shape[-1])-1) ## for NLL
        preds_NI, trues_NI = torch.LongTensor(toks_NI.shape[0], n_greedy), torch.LongTensor(toks_NI.shape[0], n_greedy) ## for decoding

        toks_BNI = torch.split(toks_NI, batch_size, dim=0) ## split in batches
        for b, batched_toks_NI in enumerate(tqdm.tqdm(toks_BNI, disable=tqdm_disable)):
            b_n = batched_toks_NI.shape[0]
            logits_NIT = model(batched_toks_NI.detach().to(model.cfg.device)) ## detach and put on device

            ## (1) NLL Metrics______________________________
            nll_NIT = modelHandlers.NegLogLik(logits_NIT.to("cpu"))
            nll_NI = nll_NIT.gather(dim=-1, index=batched_toks_NI[:, 1:, None])[:, :, 0] 
            nll_Nk, idcs_Nk = torch.topk(nll_NI, k=int(0.2*nll_NI.shape[-1]), largest=True, dim=-1) ## (2) minK

            all_nll_NI[b*batch_size:(b*batch_size)+b_n,:] = nll_NI
            minK_nll_NI[b*batch_size:(b*batch_size)+b_n,:] = nll_Nk

            ## (2) Argmax Greedy Decoding______________________________
            logits_NIT = logits_NIT[...,-(n_greedy+1):-1,:] ## only take continuations 
            top_scores_Ik, top_idcs_Ik = modelHandlers.get_topK(logits_NIT[...,:,:], topK=1, minK=False)

            preds_NI[b*batch_size:(b*batch_size)+b_n,:] = top_idcs_Ik[...,0]
            trues_NI[b*batch_size:(b*batch_size)+b_n,:] = batched_toks_NI.detach()[...,-n_greedy:]
    return (all_nll_NI, minK_nll_NI), (preds_NI, trues_NI)



def model_eval(model,c_NI:torch.LongTensor=None,c_orig_pred_NI:torch.LongTensor=None,k_NI:torch.LongTensor=None,k_orig_pred_NI:torch.LongTensor=None,I_range:list=[50,100], print_pred:bool=True):
    """
    evaluate the language model on individual batches of c_toks_NI and k_toks_NI
    """
    ## change set
    (c_mean_nll, c_minK_nll), (c_NI_pred, c_NI_true) = evaluation.evaluate_nll_greedy(model, c_NI, batch_size=50)
    if c_orig_pred_NI is not None:
        c_NI_pred = model.generate(input=c_NI[:,:50], stop_at_eos=False, max_new_tokens=50, do_sample=False)
        c_NI_pred, c_orig_pred_NI = c_NI_pred[...,I_range[0]:I_range[1]].to("cpu"), c_orig_pred_NI[...,I_range[0]:I_range[1]].to("cpu")
        c_em_N = evaluation.compute_exact_match(c_NI_pred, c_orig_pred_NI, until_wrong=False)
    else: ## argmax greedy decoding
        print("argmax greedy decoding on c_NI")
        c_em_N = evaluation.compute_exact_match(c_NI_pred, c_NI_true, until_wrong=True)

    ## keep set
    (k_mean_nll, k_minK_nll), (_, _) = evaluation.evaluate_nll_greedy(model, k_NI, batch_size=50)
    k_NI_pred = model.generate(input=k_NI[:,:50], max_new_tokens=I_range[1]-I_range[0], do_sample=False)
    k_NI_pred, k_orig_pred_NI = k_NI_pred[...,I_range[0]:I_range[1]].to("cpu"), k_orig_pred_NI[...,I_range[0]:I_range[1]].to("cpu")
    if c_orig_pred_NI is not None:
        k_em_N = evaluation.compute_exact_match(k_NI_pred, k_orig_pred_NI, until_wrong=False)
    else:
        k_em_N = evaluation.compute_exact_match(k_NI_pred, k_orig_pred_NI, until_wrong=True)

    ## process change and keep set
    c_mean_nll, k_mean_nll = round(c_mean_nll[...,I_range[0]:I_range[1]].mean().detach().item(),4), round(k_mean_nll[...,I_range[0]:I_range[1]].mean().detach().item(),4)
    
    c_changed_frac = torch.where(c_em_N == int(I_range[1]-I_range[0]), 0, 1).sum()
    k_kept_frac = torch.where(k_em_N == int(I_range[1]-I_range[0]), 1, 0).sum() 

    print(f"---Greedy EM--- change set: {c_em_N.mean().item()} [changed {c_changed_frac}/{c_em_N.shape[0]}], keep set: {k_em_N.mean().item()} [kept {k_kept_frac}/{k_em_N.shape[0]}]")
    print(f"---Mean NLL--- change set: {c_mean_nll}, keep set: {k_mean_nll}\n\n")
    
    if print_pred:
        print(f"c_NI_pred: {model.to_string(c_NI_pred)}\n")
        print(f"k_NI_pred: {model.to_string(k_NI_pred)}")
        

def compute_exact_match(preds_NI:torch.LongTensor, trues_NI:torch.LongTensor, until_wrong:bool=False):
    """
    computing the exact token match from left to right
    """
    em_NI = torch.where(preds_NI==trues_NI, 1, 0)
    em_NI = torch.cat([em_NI,torch.zeros(em_NI.shape[0],1)], dim=-1) ## add zero at the end
    if until_wrong: ## count exact match until first wrong prediction
        em_counts = em_NI.argmin(dim=1)
    else:
        em_counts = em_NI.sum(-1)
    return em_counts.float()


## GRADIENTS_____________________________________________________________________________________________________________


import transformer_lens, torch, sys, tqdm, collections
sys.path.append('/home/jupyter/')
from paraMem.utils import modelHandlers, localizing


def pool_tensor(orig_tensor:torch.tensor, dims:list, match_size:tuple=None):
    if dims is not None:
        for dim in dims:
            orig_tensor_pooled = orig_tensor.mean(dim)
            if match_size is None:
                orig_tensor = torch.repeat_interleave(orig_tensor_pooled.unsqueeze(dim), orig_tensor.shape[dim], dim=dim)
            else: ## expand the size of pooled tensor dimension to match that of match_size
                orig_tensor = torch.repeat_interleave(orig_tensor_pooled.unsqueeze(dim), match_size[dim], dim=dim)
    return orig_tensor


def clip_grads(model, min_grad:float=None, full_remove_idcs:list=[]): ## in-place
    """
    clip gradients than are not above min_grad, and not below min_grad if keep_neg is enabled
    """
    #param_vec = torch.nn.utils.parameters_to_vector(model.parameters())
    #torch.nn.utils.vector_to_parameters(param_vec, model.parameters())
    print(f"clipped at {min_grad} or using full_remove_idcs on {len(full_remove_idcs)} modules")
    list_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            if min_grad is not None:
                remove_idcs = torch.where((param.grad > min_grad) | (param.grad < -min_grad), 0, 1)
            else:
                remove_idcs = full_remove_idcs[list_idx]
                list_idx += 1
            param.grad[remove_idcs.bool()] = 0.0 ## annul small positive and negative grads
            if min_grad is not None:
                full_remove_idcs.append(remove_idcs)
    return full_remove_idcs


def contrast_metric(c_nll:torch.tensor, k_nll:torch.tensor=None, k_nll_fixed:torch.tensor=None, I_range:list=[0,100], norm_sets:float=None, pool:dict={"c": [-1], "k": [0,-1]}):
    """
    minimizing / preserve keep_score while maximizing change_score
    """
    ## (1) preprocess________________________________________
    ## select latter tokens to apply metric to
    c_nll, k_nll = c_nll[...,I_range[0]:I_range[1]], k_nll[...,I_range[0]:I_range[1]]
        
    ## (2) pooling_______________________________________________
    ## pool over dims but then expand again to retain shapes
    c_nll = pool_tensor(c_nll, pool["c"], match_size=k_nll.shape) 
    k_nll = pool_tensor(k_nll, pool["k"], match_size=None)
                
    ## adjust shapes
    ## clip batch sizes and paragraph lengths always to shorter version
    c_nll = c_nll[:k_nll.shape[0], :k_nll.shape[1]]
    k_nll = k_nll[:c_nll.shape[0], :c_nll.shape[1]]
    
    #if norm_sets: ## balance out loss terms  
    #    c_nll = torch.nn.functional.normalize(c_nll, p=1.0, dim=-1)
    #    k_nll = torch.nn.functional.normalize(k_nll, p=1.0, dim=-1)
    print(f"pooling c_nll {c_nll.shape}, k_nll {k_nll.shape} pool: {pool}")
    
    ## (3) apply metric_______________________________________________
    if isinstance(norm_sets, float):
        c_nll = c_nll * (norm_sets*(k_nll.detach().sum() / c_nll.detach().sum())) 
        #c_nll = c_nll * norm_sets
    if k_nll_fixed is not None: ## mean squared error version to enforce non-changing keep set NLL
        k_nll_fixed = k_nll_fixed[...,I_range[0]:I_range[1]]
        k_nll_fixed = pool_tensor(k_nll_fixed, pool["k"], match_size=None)
        k_nll_fixed = k_nll_fixed[:k_nll.shape[0], :k_nll.shape[1]]
        k_mse = (k_nll-k_nll_fixed.detach())**2
        contrast_res = (k_mse - c_nll).mean()
        print(f"contrast loss: {contrast_res}, c_nll: {c_nll.detach().mean()}, k_nll_mse: {k_mse.detach().mean()}")
    else:
        contrast_res = (k_nll - c_nll).mean()
        print(f"contrast loss: {contrast_res}, c_nll: {c_nll.detach().mean()}, k_nll: {k_nll.detach().mean()}")
    return contrast_res, None



def add_fwd_bwd_hooks(model, hook_filter:dict={"not in":"_input"}):
    """
    adding hooks to model to store activations in forward and backward pass
    """
    if "not in" in hook_filter.keys():
        hook_map = lambda name: hook_filter["not in"] not in name 
    elif "is in" in hook_filter.keys():
        hook_map = lambda name: hook_filter["is in"] in name
    else:
        hook_map = lambda name: "" in name 

    model.reset_hooks()
    fwd_cache = {}

    def forward_cache_hook(act, hook):
        fwd_cache[hook.name] = act.detach()
    model.add_hook(hook_map, forward_cache_hook, "fwd")

    bwd_cache = {}
    def backward_cache_hook(act, hook):
        bwd_cache[hook.name] = act.detach()
    model.add_hook(hook_map, backward_cache_hook, "bwd")
    return fwd_cache, bwd_cache


def batched_fwd_bwd(model, dl, metric_fn, fwd_bwd_fn, n_batches:int=5, c_types:list=None, fwd:bool=False):
    """
    summing all gradient weights in component c_type over multiple batches
    """
    weight_gradients = collections.defaultdict(torch.tensor)
    for batch_i, (c_toks_NI, k_toks_NI) in tqdm.tqdm(enumerate(dl)):
        fwd_cache, bwd_cache, topk_idcs = fwd_bwd_fn(model, metric_fn=metric_fn, c_toks_NI=c_toks_NI.squeeze(0), k_toks_NI=k_toks_NI.squeeze(0), c_types=c_types)
        
        if topk_idcs is None and c_types is not None : ## if collect_fn is passed
            for c_type in c_types:
                if c_type in ["W_Q","W_K","W_V","W_O","W_in","W_out"]: ## model params
                    c_vals, c_names = localizing.collect_c_type(model=model, cache=None, c_type=c_type)
                elif c_type in ["q","k","v","o","mlp_in","mlp_out"]: ## activation
                    if fwd: ## get fwd cache activations
                        c_vals, c_names = localizing.collect_c_type(model=model, cache=fwd_cache, c_type=c_type)
                    else: ## get bwd cache activations
                        c_vals, c_names = localizing.collect_c_type(model=model, cache=bwd_cache, c_type=c_type)
                
                ## Summing up values___________________________
                c_vals = c_vals.detach() / n_batches
                if batch_i==0:
                    weight_gradients[c_type] = c_vals
                else:
                    weight_gradients[c_type] += c_vals
        else:
            weight_gradients[batch_i] = topk_idcs
            
        if batch_i+1 == n_batches: ## break early
            return weight_gradients
    return weight_gradients


def run_fwd_bwd(model, metric_fn, c_toks_NI:torch.LongTensor=None, k_toks_NI:torch.LongTensor=None, optim_step:bool=False, topK:float=None, grad_norm:float=None, c_types:list=None):
    """
    adding hooks to model, running model on data on metric and returning cached activs, params are cached in model
    """
    assert len(c_toks_NI.shape)==2, f"c_toks_NI has too many dims {len(c_toks_NI.shape)}"
    
    ## refresh grads and activation caches
    model.zero_grad()
    fwd_cache, bwd_cache = add_fwd_bwd_hooks(model, hook_filter={"not in":"_input"})
    c_nll = modelHandlers.gather_token_scores(modelHandlers.NegLogLik(model(c_toks_NI.to(model.cfg.device))).to("cpu"), c_toks_NI)
    if k_toks_NI is not None: ## may only want to pass a single set of tokens
        k_nll = modelHandlers.gather_token_scores(modelHandlers.NegLogLik(model(k_toks_NI.to(model.cfg.device))).to("cpu"), k_toks_NI)    
        metric_res, metric_norm = metric_fn(c_nll, k_nll)
    metric_res, metric_norm = metric_fn(c_nll)
    
    metric_res.backward(retain_graph=False)
    fwd_cache = transformer_lens.ActivationCache(fwd_cache, model)
    bwd_cache = transformer_lens.ActivationCache(bwd_cache, model)
    
    if grad_norm is not None:
        print(f"applied grad norm clipping with max norm {grad_norm}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_norm), norm_type=2.0)
               
    topk_idcs = None
    if topK is not None:
        min_grad, topk_idcs = localizing.find_topK_grads(model, topK=topK, c_types=c_types, abs_grad=False)
        clip_grads(model, min_grad, keep_neg=False)
    
    if optim_step and hasattr(model, 'optim'):
        print(f"optimizer step")
        model.optim.step()
        model.optim.zero_grad()         
    return fwd_cache, bwd_cache, topk_idcs



## INTERVENING_______________________________________________________________________________________________________________

import torch, tqdm, collections, copy
import numpy as np
from pathlib import Path

import sys
sys.path.append('/home/jupyter/')
from paraMem.utils import localizing, gradient, modelHandlers

def find_topK_grads(model, topK:float=0.001, c_types:list=["W_in", "W_out", "W_K", "W_V", "W_Q", "W_O"]):
    """
    find the topK weights in all model parameters
    """
    ## (1) collect all params
    all_param_grads = list()
    all_param_grads = torch.cat([torch.abs(param.grad.view(-1)) for name, param in model.named_parameters() if name.split(".")[-1] in c_types])

    ## (2) identify top params (sparsity)
    if 0.0 < topK < 1.0: ## percentage
        topk_vals_flat, topk_idcs_flat = torch.topk(all_param_grads, k=int(topK*len(all_param_grads)), largest=True)
    elif 1.0 <= topK < len(all_params): ## pick top weights
        topk_vals_flat, topk_idcs_flat = torch.topk(all_param_grads, k=int(topK), largest=True)
    min_grad = torch.min(topk_vals_flat)
    print(f"{len(topk_idcs_flat)} weights in {c_types} with grads > {min_grad.item()} abs_grad: {abs_grad}")
    return min_grad, topk_idcs_flat


def intervene_params(model, c_weights_lk:dict, std:float=1.0):
    """
    perform intervention on model params according to weight_ids_LK
    """
    n_weights, layers = 0, []
    model = modelHandlers.load_model(model) ## reloading the model
    for name, param in model.named_parameters():
        name_list = name.split(".")
        c_type = name_list[-1]
        if c_type in c_weights_lk.keys():
            l, param_shape = int(name_list[1]), param.shape
            weight_ids = torch.LongTensor(c_weights_lk[c_type]["idcs"][l])
            if len(weight_ids) > 0:
                multidim_ids = np.array(np.unravel_index(weight_ids, param_shape)).T
                multidim_ids = torch.LongTensor(multidim_ids)
        
                with torch.no_grad():   
                    set_vals = torch.normal(mean=0.0, std=torch.ones(multidim_ids.shape[0]) * std)
                    if multidim_ids.shape[1]==2:
                        param[multidim_ids[:,0], multidim_ids[:,1]] += set_vals
                    elif multidim_ids.shape[1]==3:
                        param[multidim_ids[:,0], multidim_ids[:,1], multidim_ids[:,2]] += set_vals
                    n_weights += multidim_ids.shape[0]
                    layers.append(l)
    print(f"intervened with {std} on a total of {n_weights} weights in {list(c_weights_lk.keys())} in layers {set(layers)}")
    model.cfg.intervention = {"std":std,"n_weights":n_weights,"c_types":list(c_weights_lk.keys())}
    return model


if __name__ == '__main__':
    c_top_grads = get_topK_grads(c_grads, topK=100, select_layers=[0,2], return_lk=False, largest=True)
    
    
## LOCALIZING_____________________________________________________________________________________________________________

import torch, tqdm, itertools, collections, transformer_lens
from pathlib import Path

import sys
sys.path.append('/home/jupyter/')
from paraMem.utils import modelHandlers, gradient

POOL_FN = {"l1": lambda x, dim: torch.norm(x, p=1, dim=dim),
         "l2": lambda x, dim: torch.norm(x, p=2, dim=dim),
         "frob": lambda x, dim: torch.linalg.matrix_norm(x, ord='fro'), ## toDo: issue requires 2D input
         "mean_abs": lambda x, dim: torch.mean(torch.abs(x), dim=dim),
         "mean": lambda x, dim: torch.mean(x, dim=dim),
         "max_abs": lambda x, dim: torch.max(torch.abs(x), dim=dim)[0],
         "max": lambda x, dim: torch.max(x, dim=dim)[0],
         "pass": lambda x, dim: (x)}

DIST_FN = {"cos": lambda x1, x2: torch.nn.functional.cosine_similarity(x1, x2, dim=-1),
           "sub": lambda x1, x2: x1-x2,
           "sub_abs": lambda x1, x2: torch.abs(x1-x2)}


def collect_c_type(model=None, cache=None, c_type:str="W_in"):
    """
    collecting the parameter or activation gradient of a single component type returned as tensor
    """
    if model is not None and cache is None: ## parameters
        c_names = [name for name, param in model.named_parameters() if c_type in name.split(".")]
        assert len(c_names) != 0, f"check c_type: {c_type}, none found"
        vals = torch.stack([param.grad for name, param in model.named_parameters() if name in c_names])
    elif model is not None and cache is not None: ## activations
        c_names = [transformer_lens.utils.get_act_name(c_type, layer) for layer in range(model.cfg.n_layers)]
        assert len(c_names) != 0, f"check c_type: {c_type}, none found"
        vals = torch.stack([cache[c_name] for c_name in c_names])
    
    vals = torch.swapaxes(vals, 0, 1) ## LNI --> NLI
    vals = torch.swapaxes(vals, 1, 2) ## NLI --> NIL
    assert model is not None or cache is not None, f"pass either model {model} or cache {cache}"
    print(f"returning {c_names[:2]}... of shape: {vals.shape}")
    return vals, c_names


def pool_c_weights(c_grads:dict, c_grads2:dict=None, pool:str="max", dist:str="cos", topP:float=1.0, norm_by_entries:bool=False, keep_heads:bool=True):
    """
    pass a dict of parameters (or two), get topP percent and pool them or take distance
    """
    vals, x = [], []
    for c_type, grads in c_grads.items():
        
        ## (1) extract weights per c_type____________________
        if c_type in ['W_Q', 'W_K', 'W_V', 'W_O'] and keep_heads: ## attention
            grads = grads.view(grads.shape[0], grads.shape[1], -1)
            if c_grads2 is not None: ## consider second
                grads2 = c_grads2[c_type]
                grads2 = grads2.view(grads2.shape[0], grads2.shape[1], -1)                
        else:
            grads = grads.view(grads.shape[0], 1, -1)
            if c_grads2 is not None: ## consider second
                grads2 = c_grads2[c_type]
                grads2 = grads2.view(grads2.shape[0], 1, -1) 
                
        ## (2) first compute distance, then pooling____________________
        n_params = grads.shape[-1]
        if c_grads2 is not None: ## compute pairwise distance
            grads = DIST_FN[dist](grads, grads2)
            if len(grads.shape) <= 2:
                pool, topP = "pass", 1.0

        ## do the pooling
        gradpool = pool_tensor(grads, pool, topP, norm_by_entries)
       
        ## (3) reshape for next steps____________________
        for i in range(gradpool.shape[-1]):
            vals.append(gradpool[:,i].squeeze())
            if gradpool.shape[-1] > 1:
                x.append(f"{c_type} H{i}")
            else:
                x.append(f"{c_type}")
    vals = torch.stack(vals).T
    y = list(range(vals.shape[0]))
    return vals, x, y


def pool_tensor(tensor:torch.tensor, pool:str="max", abs_vals:bool=True, topP:float=1.0, norm_by_entries:bool=False):
    """
    pool a tensor and normalize it by the number of entries
    """
    n_params = tensor.numel()
    if abs_vals: ## take absolute values
        tensor = torch.abs(tensor)
    if 0.0 < topP < 1.0:
        topP = max(int(topP*tensor.shape[-1]), 1) 
    tensor, idcs = torch.topk(tensor, int(topP), dim=-1, largest=True)  
    tensorpool = POOL_FN[pool](tensor, dim=-1) ## do pooling
    if norm_by_entries:
        tensorpool = tensorpool / ((n_params)**(1/2))
    return tensorpool


def find_topK_grads(model, topK:float=0.001, abs_grad:bool=True, c_types:list=["W_in", "W_out", "W_K", "W_V", "W_Q", "W_O"]):
    """
    find the topK weights in all model parameters
    """
    ## (1) collect all params
    all_params = list()
    all_params = torch.cat([param.grad.view(-1) for name, param in model.named_parameters() if name.split(".")[-1] in c_types])
    if abs_grad:
        all_params = torch.abs(all_params)

    ## (2) identify top params (sparsity)
    if 0.0 < topK < 1.0: ## percentage
        topk_vals, topk_idcs = torch.topk(all_params, k=int(topK*len(all_params)), largest=True)
    elif 1.0 <= topK < len(all_params): ## pick top weights
        topk_vals, topk_idcs = torch.topk(all_params, k=int(topK), largest=True)
    min_grad = torch.min(topk_vals)
    print(f"{len(topk_idcs)} weights in {c_types} with grads > {min_grad.item()} abs_grad: {abs_grad}")
    return min_grad, topk_idcs


def filter_topK_grads(c_grads:dict, topK:int=100, select_c:list=None, select_l:list=None, select_heads:list=[], return_lk:bool=False, select_random:bool=False):
    """
    weight_gradients is a list of tensors, collect topK weight gradients and return as layer-wise list and in original shape
    """
    c_grads = copy.deepcopy(c_grads)
    ## (1) prepare components and layers
    ## layer prepping
    n_layers = c_grads["W_V"].shape[0]
    if select_l is None or len(select_l) == 0:
        select_l = list(range(n_layers))
    remove_layers = list(range(n_layers))
    remove_layers = list(set(remove_layers).difference(set(select_l)))
    
    ## attention head prepping
    n_heads = c_grads["W_V"].shape[1]
    if select_heads is None or len(select_heads) == 0:
        select_heads = list(range(n_heads))
    remove_heads = list(range(n_heads))
    remove_heads = list(set(remove_heads).difference(set(select_heads)))
    
    if select_c is None or len(select_c) == 0:
        select_c = list(c_grads.keys())
        
    ## (2) gather the top gradients
    c_top_grads = {}
    for c_type,c_vals in c_grads.items():
        if c_type in select_c:
            
            if len(select_heads) > 0 and len(c_vals.shape) >= 4: ## ATTN
                gradients_LHD = c_vals.view(c_vals.shape[0],c_vals.shape[1],-1)
                gradients_LHD[:,torch.LongTensor(remove_heads),:] = gradients_LHD[:,torch.LongTensor(remove_heads),:]*0 ## ZERO OUT
                c_vals = gradients_LHD
            gradients_LD = c_vals.view(c_vals.shape[0],-1) ## flatten tensor to l_dim and model_dim

            ## filter layers based on select_layers criterion  
            gradients_LD[torch.LongTensor(remove_layers),:] = gradients_LD[torch.LongTensor(remove_layers),:]*0 
            if select_random==False: ## normal topK selection mode
                weight_scores, weight_idcs = torch.topk(gradients_LD.flatten(), topK, largest=True)
            else: ## selecting any random weights as a baseline
                random_idcs = torch.randperm(gradients_LD.flatten().shape[0])
                weight_scores, weight_idcs = gradients_LD.flatten()[random_idcs[:topK]], random_idcs[:topK].squeeze()
            weight_idcs = torch.tensor(np.array(np.unravel_index(weight_idcs.numpy(), gradients_LD.shape))).T
            c_top_grads[c_type]={"idcs": weight_idcs, "scores": weight_scores}

            if return_lk: ## reformat the output to return layer-wise list of lists
                weight_ids_lk = [[] for l in range(n_layers)]
                weight_scores_lk = [[] for l in range(n_layers)]
                for k, (weight_idx, weight_score) in enumerate(zip(weight_idcs,weight_scores)):
                    weight_ids_lk[weight_idx[0]].append(weight_idx[1].item())
                    weight_scores_lk[weight_idx[0]].append(weight_score.item())
                c_top_grads[c_type]={"idcs": weight_ids_lk, "scores": weight_scores_lk}
    return c_top_grads


def store_data(data, path:str, meta:dict={}):
    """
    storing the indices of the localized weights
    """
    data_path = Path("/home/jupyter/paraMem/") / str(path)    
    torch.save(data, data_path)
    print(f"stored data at {data_path}")
    
    
def load_data(path:str):
    """
    loading the indices of localized weights
    """
    data_path = Path("/home/jupyter/paraMem/") / str(path)
    tensor = torch.load(data_path)
    return tensor



if __name__ == '__main__':
    pass

## MODEL HANDLERS_____________________________________________________________________________________________



import torch, gc, transformer_lens, itertools, tqdm


def load_model(model_type:str="gpt2-small", DEVICE:str="cpu", lr:float=0.0, weight_decay:float=0.0):
    """
    load pretrained model (and optimizer)
    """
    if isinstance(model_type, str) == False:
        model = model_type
        DEVICE = model.cfg.device
        model_type = model.cfg.model_name
        if lr > 0.0:
            pass
        elif hasattr(model, 'optim'):
            lr = model.optim.param_groups[0]["lr"]
            weight_decay = model.optim.param_groups[0]["weight_decay"]
        print(f"reset model {model_type}")
    model = transformer_lens.HookedTransformer.from_pretrained(model_type, device=DEVICE)
    model.set_use_attn_result(True)
    model.set_use_attn_in(True)
    model.set_use_hook_mlp_in(True)
    
    ## load and add optimizer
    if lr > 0.0:
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.optim = optim
        print(f"added optimizer with lr: {lr} and weight_decay: {weight_decay}")
    set_no_grad(model) ## set no grad
    return model


def gpu_check(print_stats:bool=True):
    """
    GPU checking and cleaning
    """
    gc.collect()
    torch.cuda.empty_cache()
    if print_stats:
        conv_byte2gb = 1000000000
        t = round(torch.cuda.get_device_properties(0).total_memory / conv_byte2gb, 4)
        r = round(torch.cuda.memory_reserved(0) / conv_byte2gb, 4)
        a = round(torch.cuda.memory_allocated(0) / conv_byte2gb, 4)
        print(f"Gb total: {t}, reserved: {r}, allocated: {a}")


def set_no_grad(model, no_grad:list=["embed", "pos_embed", "unembed", "b_in", "b_out", "b_K", "b_Q", "b_V", "b_O"]):
    """
    explicitely set or remove parameters that do not require gradient
    """
    #if any("abc" in s for s in xs):
    print(f"setting no_grad on {no_grad}")
    for name, param in model.named_parameters():
        for name_part in name.split("."):
            if name_part in no_grad:
                param.requires_grad = False


def NegLogLik(logits_T:torch.FloatTensor):
    """
    standard negative log likelihood loss for language modeling
    """
    log_probs_T = -(logits_T.log_softmax(dim=-1)) ## negative log probs 
    return log_probs_T


def gather_token_scores(scores_NIT:torch.FloatTensor, labels_NI:torch.LongTensor=None):
    """
    select scores corresponding to sequence of tokens (NIT)
    """     
    if len(labels_NI.shape) == 2: ## single sequence of labels
        scores = scores_NIT.gather(dim=-1, index=labels_NI[:, 1:, None])[:, :, 0] 
    elif len(labels_NI.shape) == 3: ## mutiple sequences of labels
          scores = scores_NIT.gather(dim=-1, index=labels_NI[:, 1:])
    return scores


def batch_decode(model, toks_NI:torch.LongTensor=None, dl=None, n_batch:int=50, start_at_tok:int=50, new_toks:int=50, do_sample:bool=False):
    """
    generate new toks from a model given a prompt
    """
    if dl is None:
        dl = zip(toks_NI, toks_NI)
    preds_NI, trues_NI = [],[]
    for batch_i, (_, k_toks_NI) in tqdm.tqdm(enumerate(dl)):
        if len(k_toks_NI.shape) == 3:
            k_toks_NI = k_toks_NI.squeeze(0)
        toks_NI = k_toks_NI.detach().to(model.cfg.device) ## detach and put on device
        toks_NI_pref, toks_NI_true = toks_NI[...,:start_at_tok], toks_NI[...,-start_at_tok:]
        toks_NI_pred = model.generate(input=toks_NI_pref, max_new_tokens=new_toks, stop_at_eos=False, eos_token_id=None, do_sample=do_sample, top_k=None, top_p=None, temperature=1.0)
        toks_NI_pred = toks_NI_pred[...,-new_toks:] ## only take continuations  
        preds_NI.append(toks_NI_pred)
        trues_NI.append(toks_NI_true)
        
        if batch_i+1 == n_batch:
            break ## break early if too many items in dl
    preds_NI = torch.stack(preds_NI).view(-1,new_toks)
    trues_NI = torch.stack(trues_NI).view(-1,new_toks)
    return preds_NI, trues_NI


def get_topK(scores_NIT:torch.FloatTensor, topK:int=None, minK:bool=False):
    """
    return the topK scores and token indeces
    """
    if topK is None:
        topK = scores_NIT.shape[-1]
    scores_NIk, indc_NIk = torch.topk(scores_NIT, topK, dim=-1, largest= not minK)
    return scores_NIk, indc_NIk



if __name__ == '__main__':
    pass

## PATCHING_______________________________________________________________________________________

import torch, tqdm

import sys
sys.path.append('/home/jupyter/')
from paraMem.utils import modelHandlers, evaluation


def get_first2sec_tok(logits_NIT:torch.Tensor, prefix_NI:torch.Tensor, keepNonTop:bool=True):
    """
    pertubate sequence via first and second most likely tokens
    """
    scores_NIT = (torch.nn.functional.softmax(logits_NIT.to("cpu"), dim=-1))
    prefix_scores_NI = modelHandlers.gather_token_scores(scores_NIT, prefix_NI)
    top_scores_Ik, top_idcs_Ik = modelHandlers.get_topK(scores_NIT, topK=2, minK=False)
    
    pertubed_prefix = torch.clone(prefix_NI[:,1:]).long()
    prefixIsTop = torch.where(top_idcs_Ik[...,:-1,0] == prefix_NI[:,1:], 1, 0)
    pertubed_prefix[prefixIsTop.bool()] = top_idcs_Ik[...,:-1,1][prefixIsTop.bool()] ## pick top 2
    if keepNonTop:
        pertubed_prefix[~prefixIsTop.bool()] = top_idcs_Ik[...,:-1,0][~prefixIsTop.bool()] ## pick top 1
    
    ## add BOS token
    bos_N = prefix_NI[:,0].unsqueeze(-1)
    pertubed_prefix = torch.cat((bos_N, pertubed_prefix), dim=-1)
    return pertubed_prefix
    
def get_random_tok(prefix_NI:torch.Tensor, vocab_size:int=50257, seed:int=0): 
    """
    pertubate sequence via random tokens (vocab_size = model.cfg.d_vocab)
    """
    if seed >= 0:
        print(f"fixed torch seed {seed}")
        torch.manual_seed(seed)
    pertubed_prefix = torch.randint(0, vocab_size, prefix_NI.shape)[...,:-1]
    
    ## add BOS token
    bos_N = prefix_NI[:,0].unsqueeze(-1)
    pertubed_prefix = torch.cat((bos_N, pertubed_prefix), dim=-1)
    return pertubed_prefix


def token_patching_loop(model, toks_NI=torch.tensor, pertubed_pref_NI=torch.tensor, decode:bool=False, disable_tqdm:bool=False):
    """
    loop over all tokens in the prefix, pertubate them and measure the change in the continuation
    """
    with torch.no_grad():
        pref_NI, cont_NI, n_toks = toks_NI[:,:pertubed_pref_NI.shape[-1]], toks_NI[:,-pertubed_pref_NI.shape[-1]:], pertubed_pref_NI.shape[-1]

        nll_metric, em_metric = torch.zeros(pref_NI.shape[0], pref_NI.shape[-1]), torch.zeros(pref_NI.shape[0], pref_NI.shape[-1])
        toks_NI = torch.cat((pref_NI, cont_NI), dim=-1)
        orig_toks_nll = modelHandlers.gather_token_scores(modelHandlers.NegLogLik(model(toks_NI.to(model.cfg.device)).to("cpu")), toks_NI)

        interv_tok_pos, min_em, most_changed_preds = torch.zeros(cont_NI.shape[0]).long(), torch.ones(cont_NI.shape[0])*9999, torch.zeros(cont_NI.shape).long()
        for tok_pos in tqdm.tqdm(range(n_toks), total=n_toks, disable=disable_tqdm):

            ## (1) intervene on token at token position
            pref_NI_interv = torch.clone(pref_NI)
            pref_NI_interv[:,tok_pos] = pertubed_pref_NI[:,tok_pos]

            ## (2) generate continuation on intervened token sequence
            if decode: #[:,:prefix_NI.shape[-1]]
                pred_toks_NI = model.generate(input=pref_NI_interv, use_past_kv_cache=False, stop_at_eos=False, max_new_tokens=cont_NI.shape[-1], do_sample=False)
                pred_nll_NIT = modelHandlers.NegLogLik(model(pred_toks_NI).detach().to("cpu"))
                pred_nll_NI = modelHandlers.gather_token_scores(pred_nll_NIT, pred_toks_NI.to("cpu"))

                pred_toks_NI = pred_toks_NI[:,-cont_NI.shape[-1]:].to("cpu")
                cont_NI_test = torch.clone(cont_NI).to("cpu")


            else: ## argmax decoding
                toks_NI_interv = torch.cat((pref_NI_interv, cont_NI), dim=-1)
                pred_nll_NIT = modelHandlers.NegLogLik(model(toks_NI_interv.to(model.cfg.device)).to("cpu"))

                pred_nll_NI = modelHandlers.gather_token_scores(pred_nll_NIT, toks_NI) ## get pred NLL 
                _, pred_toks_NIk = modelHandlers.get_topK(pred_nll_NIT, topK=1, minK=True) ## get argmax toks 
                pred_toks_NI = pred_toks_NIk[...,-(cont_NI.shape[-1]+1):-1,0].to("cpu")
                cont_NI_test = torch.clone(cont_NI[...,:]).to("cpu")

            ## (3) evaluate the generated continuation against the original continuation
            nll_metric[:,tok_pos] = pred_nll_NI[:,-cont_NI.shape[-1]:].mean(-1)   

            em = evaluation.compute_exact_match(pred_toks_NI, cont_NI_test, until_wrong=False)
            em_metric[:,tok_pos] = em

            ## (4) update minimum em and most_changed_preds
            select_mask = torch.where(em < min_em, 1, 0)
            select_idcs = torch.nonzero(select_mask.bool()).squeeze()
            interv_tok_pos[select_idcs] = tok_pos
            min_em[select_idcs] = em[select_idcs]
            most_changed_preds[select_idcs,:] = pred_toks_NI[select_idcs,:].detach()        
    return nll_metric, em_metric, most_changed_preds, interv_tok_pos


def pick_intervention_tok(metric_NI:torch.tensor, largest:bool=False):
    """
    picking the token in the prefix that causes the biggest change in the continuation
    """
    if largest:
        vals, idcs = torch.max(metric_NI,dim=-1)
    else:
        vals, idcs = torch.min(metric_NI,dim=-1)
    idcs = idcs.view(metric_NI.shape[0], 1) ## reshape back to NI
    vals = vals.view(metric_NI.shape[0], 1) ## reshape back to NI
    return vals, idcs


def get_interv_impact_indeces(c_toks_NI:torch.tensor, k_toks_NI:torch.tensor):
    """
    function to get the positions of the intervention (src) and impact (trg) token
    """
    ck_diff_mask = torch.where(c_toks_NI != k_toks_NI, 1,0)
    ck_diff_cumsum = torch.cumsum(ck_diff_mask, dim=-1) ## intervention

    ## find intervention
    src_NI = (ck_diff_cumsum==1).nonzero() 
    src_idcs = torch.cat((torch.zeros(1),(src_NI[:-1,0] != src_NI[1:,0]).nonzero(as_tuple=True)[0] + 1)).long()
    src_NI = src_NI[src_idcs]

    ## find impact
    trg_NI = (ck_diff_cumsum==2).nonzero() 
    trg_NI_idcs = torch.cat((torch.zeros(1),(trg_NI[:-1,0] != trg_NI[1:,0]).nonzero(as_tuple=True)[0] + 1)).long()
    trg_NI = trg_NI[trg_NI_idcs]
    
    return src_NI, trg_NI