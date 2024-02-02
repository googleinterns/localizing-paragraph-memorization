import torch, tqdm, itertools, collections, transformer_lens, math
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
        if len(vals.shape) > 3: ## attention params with heads
            vals = torch.swapaxes(vals, 0, 1) ## LHDD --> HLDD
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
    vals, names = [], []
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
                names.append(f"{c_type} H{i}")
            else:
                names.append(f"{c_type}")
    vals = torch.stack(vals).T
    layers = list(range(vals.shape[0]))
    return vals, names, layers


def pool_tensor(tensor:torch.tensor, pool:str="max", abs_vals:bool=True, topP:float=1.0, norm_by_entries:bool=False):
    """
    pool a tensor and normalize it by the number of entries
    """
    n_params = tensor.numel()
    if len(tensor.shape) == 5: ##ATTN
        n_params = n_params / 12 ## devide by number of heads    

    if abs_vals: ## take absolute values
        tensor = torch.abs(tensor)
        
    norm_by = 1.0
    if norm_by_entries:
        norm_by = math.log(n_params)#n_params**(1/2)
        tensor[tensor!=0] = tensor[tensor!=0]*(1/norm_by)
    
    if 0.0 < topP < 1.0:
        topP = max(int(topP*tensor.shape[-1]), 1) 
    topK_vals, topK_idcs = torch.topk(tensor, int(topP), dim=-1, largest=True)  
    tensorpool = POOL_FN[pool](topK_vals, dim=-1) ## do pooling
    
    #print(f"abs_vals {abs_vals}, topP {topP} selected, {pool} pooled and normalized by: {norm_by}")
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

