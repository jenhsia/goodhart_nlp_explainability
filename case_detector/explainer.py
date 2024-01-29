# import lime
import numpy as np
import torch
import json
from datasets import load_dataset
# from lime_explainer import LimeExplainer
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pdb
import pickle
import torch.nn as nn
from lime.lime_text import LimeTextExplainer
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer, LayerIntegratedGradients, LayerDeepLift, ShapleyValueSampling, GradientShap, KernelShap, FeatureAblation, LayerConductance

def get_tokenizer_out(tokenizer, text, query, device):
    tokenized_text= np.array(tokenizer.convert_tokens_to_ids(text.split()))
    text_len = len(tokenized_text)
    tokenized_query = np.array(tokenizer.convert_tokens_to_ids(query.split()))
    query_len = len(tokenized_query)
    tokenizer_out = dict()
    max_text_len = (int)(512/2 -2)
    max_query_len = (int)(512/2 -1)
    tokenizer_out['input_ids'] = np.zeros((1, 512))
    tokenizer_out['token_type_ids'] = np.zeros((1, 512))
    tokenizer_out['attention_mask'] = np.zeros((1, 512))
    if query == None or len(query) == 0 :
        trunc_text_len = np.min(text_len, max_text_len)
        trunc_query_len = 0
        tokenizer_out['input_ids'][0][0] = tokenizer.cls_token_id
        tokenizer_out['input_ids'][0][1:1+trunc_text_len] = tokenized_text[:trunc_text_len]
        tokenizer_out['input_ids'][0][1+trunc_text_len] = tokenizer.sep_token_id
        tokenizer_out['attention_mask'][0][:trunc_text_len] = 1
    else:
        if (text_len+query_len <= (max_text_len+max_query_len)):
            trunc_text_len = text_len
            trunc_query_len= query_len
        elif (text_len > max_text_len  and query_len  > max_query_len):
            trunc_text_len = max_text_len
            trunc_query_len = max_query_len
        elif (text_len < max_text_len and query_len >= max_query_len):
            trunc_text_len = text_len
            trunc_query_len = max_query_len + max_text_len - text_len
        elif (text_len >= max_text_len and query_len < max_query_len):
            trunc_text_len = max_query_len + max_text_len - query_len
            trunc_query_len = query_len
        
        tokenizer_out['input_ids'][0][0] = tokenizer.cls_token_id
        tokenizer_out['input_ids'][0][1:1+trunc_text_len] = tokenized_text[:trunc_text_len]
        tokenizer_out['input_ids'][0][1+trunc_text_len] = tokenizer.sep_token_id
        tokenizer_out['input_ids'][0][2+trunc_text_len:2+trunc_text_len + trunc_query_len] = tokenized_query[:trunc_query_len]
        tokenizer_out['input_ids'][0][2+trunc_text_len + trunc_query_len] = tokenizer.sep_token_id
        tokenizer_out['token_type_ids'][0][trunc_text_len+2:] = 1
        tokenizer_out['attention_mask'][0][:text_len+query_len+3] = 1
    tokenizer_out['input_ids'] = torch.tensor(tokenizer_out['input_ids']).type(torch.LongTensor).to(device)
    tokenizer_out['token_type_ids'] = torch.tensor(tokenizer_out['token_type_ids']).type(torch.LongTensor).to(device)
    tokenizer_out['attention_mask'] = torch.tensor(tokenizer_out['attention_mask']).type(torch.LongTensor).to(device)
    return tokenizer_out, trunc_text_len

def combine_indices(indices):
    hard_rationale_predictions = []
    if(len(indices) == 0):
        hard_rationale_predictions.append({"end_token": (int)(0), "start_token": (int)(0)})
        return hard_rationale_predictions
    start_index = indices[0]
    prev_index = start_index
    if(len(indices)==1):
        hard_rationale_predictions.append({"end_token": (int)(start_index+1), "start_token": (int)(start_index)})
        return hard_rationale_predictions
    for next_index in indices[1:]:
        if (next_index != (prev_index +1)):
            end_index = prev_index +1
            hard_rationale_predictions.append({"end_token": (int)(end_index), "start_token": (int)(start_index)})
            start_index = next_index
        prev_index = next_index
    if (prev_index == indices[-1]):
        hard_rationale_predictions.append({"end_token": (int)(prev_index +1), "start_token": (int)(start_index)})
    return hard_rationale_predictions

class RandomExplainer:
    def __init__(self, top_k, model, tokenizer):
        self.k = top_k
        self.tokenizer = tokenizer
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print('self device', self.device)
        self.model = model
        self.model.to(self.device)

    def get_top_k_rat_and_removed(self, text, query):
        top_k_mask, soft_rationales, tokenizer_out, trunc_text_len = self.get_top_k_mask(text, query)
        top_k_mask = np.array(top_k_mask[0])
        soft_rationales = np.concatenate([soft_rationales, np.zeros(len(text.split()) - len(soft_rationales))])
        rat_only_indices = np.where(top_k_mask == 1.0)[0]
        rat_only_indices = combine_indices(rat_only_indices)
        tokens = np.array(text.split())[:trunc_text_len]
        rationale_text = (' ').join(tokens[top_k_mask.astype(bool)])
        rationale_removed_text = (' ').join(tokens[(1 -(top_k_mask)).astype(bool)])
        
        return rationale_text, rationale_removed_text, rat_only_indices, soft_rationales

    def get_top_k_mask(self, text, query):
        tokenizer_out, trunc_text_len = get_tokenizer_out(self.tokenizer, text, query, self.device)
        num_words = int(0.01 * self.k * trunc_text_len)
        num_words = 1 if num_words == 0 else num_words
        score = torch.rand(trunc_text_len)
        selected_indices = torch.argsort(score, descending=True)[:num_words]
        top_k_mask = [1.0 if i in selected_indices else 0.0 for i in range(trunc_text_len)]
        return [top_k_mask], score, tokenizer_out, trunc_text_len


class AttentionExplainer:
    def __init__(self, top_k, model, tokenizer):
        self.k = top_k
        self.tokenizer = tokenizer
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print('self device', self.device)
        self.model = model
        self.model.to(self.device)
    
    def get_top_k_rat_and_removed(self, text, query):
        pdb.set_trace()
        top_k_mask, soft_rationales, tokenizer_out, trunc_text_len = self.get_top_k_mask(text, query)
        top_k_mask = np.array(top_k_mask[0])
        soft_rationales = np.concatenate([soft_rationales, np.zeros(len(text.split()) - len(soft_rationales))])
        rat_only_indices = np.where(top_k_mask == 1.0)[0]
        rat_only_indices = combine_indices(rat_only_indices)
        tokens = np.array(text.split())[:trunc_text_len]
        rationale_text = (' ').join(tokens[top_k_mask.astype(bool)])
        rationale_removed_text = (' ').join(tokens[(1 -(top_k_mask)).astype(bool)])
        pdb.set_trace()
        return rationale_text, rationale_removed_text, rat_only_indices, soft_rationales
    
    def get_top_k_mask(self, text, query):
        tokenizer_out, trunc_text_len = get_tokenizer_out(self.tokenizer, text, query, self.device)
        model_out = self.model(**tokenizer_out)
        attentions = torch.stack(model_out.attentions)
        attention_scores = attentions[-1].sum(dim = 1)[:, 0,1:1+trunc_text_len]
        output_mask = []

        num_words = int(0.01 * self.k * trunc_text_len)

        num_words = 1 if num_words == 0 else num_words
        attention_score = attention_scores[0]
        selected_indices = torch.argsort(attention_score, descending=True)[:num_words]
        top_k_mask = [1.0 if i in selected_indices else 0.0 for i in range(trunc_text_len)]
        output_mask.append(top_k_mask)
        return output_mask, attention_score.cpu().detach().numpy(), tokenizer_out, trunc_text_len
    
class IntegratedGradientExplainer:

    def __init__(self, top_k, model, tokenizer):
        self.k = top_k
        self.tokenizer = tokenizer
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print('self device', self.device)
        # self.model = nn.DataParallel(model)
        self.model = model
        self.model.to(self.device)
        self.lig = LayerIntegratedGradients(model, model.bert.embeddings)

    
    def get_top_k_rat_and_removed(self, text, query):
        pdb.set_trace()
        top_k_mask, soft_rationales, tokenizer_out, trunc_text_len = self.get_top_k_mask(text, query)
        top_k_mask = np.array(top_k_mask[0])
        # pdb.set_trace()
        soft_rationales = np.concatenate([soft_rationales, np.zeros(len(text.split()) - len(soft_rationales))])
        rat_only_indices = np.where(top_k_mask == 1.0)[0]
        rat_only_indices = combine_indices(rat_only_indices)
        tokens = np.array(text.split())[:trunc_text_len]
        rationale_text = (' ').join(tokens[top_k_mask.astype(bool)])
        rationale_removed_text = (' ').join(tokens[(1 -(top_k_mask)).astype(bool)])
        return rationale_text, rationale_removed_text, rat_only_indices, soft_rationales
        pdb.set_trace()
    def get_top_k_mask(self, text, query):
        tokenizer_out, trunc_text_len = get_tokenizer_out(self.tokenizer, text, query, self.device)
        output = self.model(**tokenizer_out)
        predictions = (torch.argmax(output, dim=-1))
        prediction = predictions[0]
    
        output_mask = []
        soft_rationales = np.zeros(trunc_text_len)
        num_words = int(0.01 * self.k * trunc_text_len)
        num_words = 1 if num_words == 0 else num_words
        inputs = (tokenizer_out['input_ids'], tokenizer_out['attention_mask'], tokenizer_out['token_type_ids'])
        
        baseline_input_ids = tokenizer_out['input_ids'].clone()
        baseline_input_ids[0][1: 1+trunc_text_len] = 0
        
        baselines = (baseline_input_ids, tokenizer_out['attention_mask'], tokenizer_out['token_type_ids'])
        attributions = self.lig.attribute(inputs=inputs, baselines=baselines, target=prediction.item(), internal_batch_size=8)

        attributions = attributions.sum(dim=-1).squeeze(0)
        score = attributions / torch.norm(attributions)
        score = score[1:trunc_text_len+1]

        selected_indices = torch.argsort(score, descending=True)[:num_words]
        top_k_mask = [1.0 if i in selected_indices else 0.0 for i in range(trunc_text_len)]
        output_mask.append(top_k_mask)
        return output_mask, score.cpu(), tokenizer_out, trunc_text_len


class LimeExplainer:

    def __init__(self, top_k, model, tokenizer):
        self.k = top_k
        self.tokenizer = tokenizer
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print('self device', self.device)
        self.model = model
        self.model.to(self.device)
        self.explainer = LimeTextExplainer(class_names=["NEG", "POS"])
        self.query_ids = ''

    
    
    def get_top_k_rat_and_removed(self, text, query):
        top_k_mask, soft_rationales, tokenizer_out, trunc_text_len = self.get_top_k_mask(text, query)
        top_k_mask = np.array(top_k_mask[0])
        soft_rationales = np.concatenate([soft_rationales, np.zeros(len(text.split()) - len(soft_rationales))])
        rat_only_indices = np.where(top_k_mask == 1.0)[0]
        rat_only_indices = combine_indices(rat_only_indices)
        tokens = np.array(text.split())[:trunc_text_len]
        rationale_text = (' ').join(tokens[top_k_mask.astype(bool)])
        rationale_removed_text = (' ').join(tokens[(1 -(top_k_mask)).astype(bool)])
        return rationale_text, rationale_removed_text, rat_only_indices, soft_rationales
    
    def get_top_k_mask(self, text ,query):
        tokenizer_in = ((text,) if query == None else (text, query))
        tokenizer_out = self.tokenizer(*tokenizer_in, padding='max_length', max_length=512, truncation=True, return_tensors = 'pt')
        tokenizer_out.to(self.device)
        output = self.model(**tokenizer_out)
        predictions = torch.argmax(output, dim=-1)
        prediction = predictions[0]
        tokenizer_out, trunc_text_len = get_tokenizer_out(self.tokenizer, text, query, self.device)
        output_mask = []
        token_ids = tokenizer_out['input_ids'][0][1:trunc_text_len+1]
        self.query = query

        num_words = int(0.01 * self.k * trunc_text_len)

        num_words = 1 if num_words == 0 else num_words

        text = " ".join(self.tokenizer.convert_ids_to_tokens(token_ids))

        num_features = min(num_words * 2, trunc_text_len)
        exp = self.explainer.explain_instance(text, self.predict_prob_batch, num_features=num_features, num_samples=2*trunc_text_len)
        selected_indices = []
        soft_rationales = np.zeros(trunc_text_len)
        for word, value in exp.as_list():
            word_id = self.tokenizer.convert_tokens_to_ids(word)
            word_occurences = np.where(token_ids.cpu() == word_id)[0]
            if prediction.item() == 1 and value < 0:
                soft_rationales[word_occurences] = 0
                # ignore negative features for positive prediction
                continue
            if prediction.item() == 0 and value > 0:
                soft_rationales[word_occurences] = 0
                # ignore positive features for negative prediction
                continue
            
            selected_indices.extend(word_occurences)
            soft_rationales[word_occurences] = value if prediction.item() == 1 else -value
            if len(selected_indices) >= num_words:
                break
        
        selected_indices = selected_indices[:num_words]
        top_k_mask = [1.0 if i in selected_indices else 0.0 for i in range(trunc_text_len)]
        output_mask.append(top_k_mask)
        return output_mask, soft_rationales, tokenizer_out, trunc_text_len

    
    def predict_prob(self, text):
        tokenizer_out, _= get_tokenizer_out(self.tokenizer, text, self.query, self.device)
        output = self.model(**tokenizer_out)
        output = torch.nn.functional.softmax(output, dim=1)[0]
        return output.detach().cpu().numpy()

    def predict_prob_batch(self, inputs):
        output = [self.predict_prob(i) for i in inputs]
        return np.array(output)