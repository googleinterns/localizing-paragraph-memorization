
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






