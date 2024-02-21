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


