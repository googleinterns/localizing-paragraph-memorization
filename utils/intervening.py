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