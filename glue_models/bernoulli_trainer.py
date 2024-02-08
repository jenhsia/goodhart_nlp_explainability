import torch
from transformers import Trainer
import math
import sys
import pdb
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


class BernoulliTrainer(Trainer):
    
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            non_padded_doc_len = [(int)(torch.where(input_ids == 102)[0][0])-1 for input_ids in data['input_ids']]
            
            bernoulli_param = [0.5*torch.ones(len) for len in non_padded_doc_len]
            bernoulli_mask = [torch.bernoulli(param) for param in bernoulli_param]
            
            masked_indices = [torch.where(mask==1)[0] for mask in bernoulli_mask]
            
            new_input_ids = torch.zeros_like(data['input_ids'])
            new_attention_mask = torch.zeros_like(data['attention_mask'])
            new_token_type_ids = torch.zeros_like(data['token_type_ids'])
            
            query_len = []
            
            for i, mask_ind in enumerate(masked_indices):
                query_len = int((data['attention_mask'][i] == 1).sum().detach().numpy())- non_padded_doc_len[i]-2
                new_doc_len = mask_ind.shape[0]
                
                new_input_ids[i][0] = 101
                new_input_ids[i][1:new_doc_len+1] = data['input_ids'][i][1+mask_ind.numpy()]
                new_input_ids[i][new_doc_len+1] = 102
                new_input_ids[i][new_doc_len+2: new_doc_len+2+query_len] = data['input_ids'][i][2+non_padded_doc_len[i]:2+non_padded_doc_len[i]+query_len]
                
                new_attention_mask[i][:new_doc_len+2+query_len] = 1
                new_token_type_ids[i][new_doc_len+2: new_doc_len+2 + query_len] = 1
                
            data['input_ids'] = new_input_ids
            data['attention_mask'] = new_attention_mask
            data['token_type_ids'] = new_attention_mask
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data