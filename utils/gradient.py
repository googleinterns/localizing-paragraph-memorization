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


import transformer_lens, torch, sys, tqdm, collections
sys.path.append('/home/jupyter/')
from paraMem.utils import modelHandlers, localizing, intervening
KLDiv = torch.nn.KLDivLoss(reduction="batchmean")


def pool_tensor(orig_tensor:torch.tensor, dims:list, match_size:tuple=None):
    """
    pooling tensors over specified dims, then expanding again to match prior size or that of match_size
    """
    if dims is not None:
        for dim in dims:
            orig_tensor_pooled = orig_tensor.mean(dim)
            if match_size is None:
                orig_tensor = torch.repeat_interleave(orig_tensor_pooled.unsqueeze(dim), orig_tensor.shape[dim], dim=dim)
            else: ## expand the size of pooled tensor dimension to match that of match_size
                orig_tensor = torch.repeat_interleave(orig_tensor_pooled.unsqueeze(dim), match_size[dim], dim=dim)
    return orig_tensor


def clip_grads(model, min_grad:float=None, full_remove_idcs:list=[], topK=0.0): ## in-place
    """
    clip gradients than are not above min_grad, and not below min_grad if keep_neg is enabled
    """
    #param_vec = torch.nn.utils.parameters_to_vector(model.parameters())
    #torch.nn.utils.vector_to_parameters(param_vec, model.parameters())
    list_idx = 0
    removed_n_weights = 0
    for param in model.parameters():
        if param.requires_grad:
            if min_grad is not None:
                if -1.0 < topK < 0.0:
                    remove_idcs = torch.bernoulli(torch.ones(param.grad.shape)*(1.0-abs(topK)))
                else:  
                    remove_idcs = torch.where((param.grad >= min_grad) | (param.grad <= -min_grad), 0, 1)
                full_remove_idcs.append(remove_idcs)
            else:
                remove_idcs = full_remove_idcs[list_idx]
                list_idx += 1
            removed_n_weights += ((~(remove_idcs.bool())).int()).sum()
            param.grad[remove_idcs.bool()] = 0.0 ## annul small positive and negative grads
            
    print(f"clipped at {min_grad} / kept {removed_n_weights.sum()} weights")
    return full_remove_idcs



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


## Single Objectives_______________________________________________________________

def run_single_fwd_bwd(model, metric_fn, c_toks_NI:torch.LongTensor, optim_step:bool=False, topK:float=None, grad_norm:float=None, c_types:list=None):
    """
    adding hooks to model, running model on data on metric and returning cached activs, params are cached in model
    """
    assert len(c_toks_NI.shape)==2, f"c_toks_NI has too many dims {len(c_toks_NI.shape)}"
    
    ## refresh grads and activation caches
    model.zero_grad()
    fwd_cache, bwd_cache = add_fwd_bwd_hooks(model, hook_filter={"not in":"_input"})
    
    c_nll = modelHandlers.gather_token_scores(modelHandlers.NegLogLik(model(c_toks_NI.to(model.cfg.device))).to("cpu"), c_toks_NI)
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



def single_seq_metric(nll_NI:torch.tensor, NI_idcs:torch.tensor=None, pool:dict={"c": []}):
    """
    minimizing / preserve keep_score while maximizing change_score
    """
    ## (1) preprocess________________________________________
    ## select tokens to apply metric to
    if isinstance(NI_idcs, torch.LongTensor):
        nll_NI = nll_NI[NI_idcs[:,0], NI_idcs[:,1]]
    elif isinstance(NI_idcs, list):
        nll_NI = nll_NI[...,NI_idcs[0]:NI_idcs[1]]
        
    ## (2) pooling_______________________________________________
    ## pool over dims but then expand again to retain shapes
    nll_NI = pool_tensor(nll_NI, pool["c"])             
    print(f"pooling nll_NI {nll_NI.shape}, pool: {pool}")
    
    ## (3) apply metric_______________________________________________
    metric_res = nll_NI.mean()
    print(f"contrast loss: {metric_res}")
    return metric_res, None


## Contrastive Objectives_______________________________________________________________

def contrast_metric(c_nll_NIT, c_toks_NI, c_perturb_toks_NI, k_logits_NIT, k_logits_fixed_NIT, I_range:list=[49,99], use_perturb:bool=True, c_set_norm:float=None):
    """
    minimizing / preserve keep_score while maximizing change_score
    """
    if use_perturb:
        c_nll_NI = modelHandlers.gather_token_scores(c_nll_NIT, c_perturb_toks_NI)
        #c_perturb_nll_NI = modelHandlers.gather_token_scores(c_nll_NIT, c_perturb_toks_NI)
        ##c_perturb_nll_NI = c_perturb_nll_NI[...,I_range[0]:I_range[1]]
    else:
        c_nll_NI = modelHandlers.gather_token_scores(c_nll_NIT, c_toks_NI)

    
    c_nll_Nc = c_nll_NI[...,I_range[0]:I_range[1]]
    k_logits_NcT, k_logits_fixed_NcT = k_logits_NIT[...,I_range[0]:I_range[1],:], k_logits_fixed_NIT[...,I_range[0]:I_range[1],:]
        
    keep = KLDiv(torch.nn.functional.log_softmax(k_logits_NcT,  dim=-1), torch.nn.functional.softmax(k_logits_fixed_NcT.detach(), dim=-1)).mean()
    
    if use_perturb:
        #change = c_set_norm *(c_perturb_nll_NI + (-c_nll_Nc)).mean()
        change = c_set_norm * (c_nll_Nc.mean())
    else:
        change = c_set_norm * (-c_nll_Nc.mean())
        
    contrast_res = (keep+change)
        
    print(f"loss: {contrast_res}, mem: {change.detach()}, non mem: {keep.detach()}, use_perturb: {use_perturb}, c_set_norm: {c_set_norm}")
    return contrast_res, None


def run_contrastive_fwd_bwd(model, metric_fn, c_toks_NI, c_perturb_toks_NI, k_toks_NI, optim_steps:int=-1, topK:float=None, grad_norm:float=None, c_types:list=None):
    """
    adding hooks to model, running model on data on metric and returning cached activs, params are cached in model
    """
    fwd_cache, bwd_cache = add_fwd_bwd_hooks(model, hook_filter={"not in":"_input"})     
    c_toks_NI = c_toks_NI.to(model.cfg.device)
    c_perturb_toks_NI = c_perturb_toks_NI.to(model.cfg.device)
    k_toks_NI = k_toks_NI.to(model.cfg.device)
    k_logits_fixed_NIT = model(k_toks_NI)

    for step_i in range(abs(optim_steps)):
        
        c_nll_NIT = modelHandlers.NegLogLik(model(c_toks_NI))
        k_logits_NIT = model(k_toks_NI)

        metric_res, metric_norm = metric_fn(c_nll_NIT, c_toks_NI, c_perturb_toks_NI, k_logits_NIT, k_logits_fixed_NIT) 

        model.zero_grad()
        metric_res.backward(retain_graph=False)

        if grad_norm is not None:
            print(f"applied grad norm clipping with max norm {grad_norm}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_norm), norm_type=2.0)

        if topK is not None:
            if step_i == 0:
                min_grad, topk_idcs = intervening.find_topK_grads(model, topK=topK, c_types=c_types)
                full_remove_idcs = clip_grads(model, min_grad, full_remove_idcs=[], topK=topK)
            full_remove_idcs = clip_grads(model, min_grad=None, full_remove_idcs=full_remove_idcs)
        else:
            topk_idcs = None

        if optim_steps >= 1 and hasattr(model, 'optim'):
            print(f"{step_i+1}/{abs(optim_steps)}, optimizer step")
            model.optim.step()
            model.optim.zero_grad()
        
    del c_toks_NI
    del c_perturb_toks_NI
    del k_toks_NI
    torch.cuda.empty_cache()

    fwd_cache = transformer_lens.ActivationCache(fwd_cache, model)
    bwd_cache = transformer_lens.ActivationCache(bwd_cache, model)
    return fwd_cache, bwd_cache, topk_idcs





