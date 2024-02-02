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
                pred_toks_NI = model.generate(input=pref_NI_interv, use_past_kv_cache=True, stop_at_eos=False, max_new_tokens=cont_NI.shape[-1], do_sample=False)
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