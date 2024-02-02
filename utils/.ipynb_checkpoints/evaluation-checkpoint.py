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


