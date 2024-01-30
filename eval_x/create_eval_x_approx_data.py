import numpy as np
import json
import os
import sys
import pdb
from collections import Counter
import operator
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rationale_benchmark.utils import load_jsonl
from datasets import load_dataset
import pickle


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
        # if not contiguous
        if (next_index != (prev_index +1)):
            end_index = prev_index +1
            hard_rationale_predictions.append({"end_token": (int)(end_index), "start_token": (int)(start_index)})
            start_index = next_index
        prev_index = next_index
    if (prev_index == indices[-1]):
        hard_rationale_predictions.append({"end_token": (int)(prev_index +1), "start_token": (int)(start_index)})
    return hard_rationale_predictions


def get_rationale(doc, pred_label, model, tokenizer, token_limit = 1):
    text = doc['text']
    query = doc['query']
    split_text = np.array(text.split())
    token_count = 0
    rat_text = []
    rat_indices = []
    soft_rationales = np.zeros(len(split_text))
    for i, token in enumerate(split_text):
        tokenizer_out = tokenizer(*(token, query), padding='max_length', max_length=512, truncation=True, return_tensors = 'pt')
        output = model(**tokenizer_out)
        eval_x_pred_label = np.argmax(output.logits.detach().numpy(), axis = 1)
        if (eval_x_pred_label == pred_label):
            soft_rationales[i] = 1
            rat_text.append(token)
            rat_indices.append(i)
            token_count +=1 
        if (token_count == token_limit):
            break
    if (token_count != 0):
        return " ".join(rat_text), np.array(rat_indices), soft_rationales
    return text, np.arange(len(split_text)), np.ones(len(split_text))

def get_top_phrase_rationale(doc, pred_label, model, tokenizer, token_limit = 1, phrase_len = 1):
    text = doc['text']
    query = doc['query']
    split_text = np.array(text.split())
    rat_text = []
    rat_indices = []
    soft_rationales = np.zeros(len(split_text))
    tokenizer_in_text = []
    
    for i in range(len(split_text)-phrase_len+1): 
        tokenizer_in_text.append(' '.join(split_text[i:i+phrase_len]))
    
    tokenizer_out = tokenizer(tokenizer_in_text, [query]*(len(split_text)-phrase_len+1), padding='max_length', max_length=len(query.split())+ 3+1, truncation = True, return_tensors = 'pt')
    output = model(**tokenizer_out)
    normalized_output = np.array([np.exp(l)/np.exp(l).sum() for l in output.logits.detach().numpy()])
    
    max_inds = np.argpartition(normalized_output[:, pred_label], -token_limit)[-token_limit:]
    rat_indices = np.sort(max_inds)
    phrase_indices = []
    for i in max_inds: 
        phrase_indices.extend(list(range(i, i + phrase_len)))

    phrase_indices = list(set(phrase_indices))
    rat_text = split_text[phrase_indices]
    soft_rationales[phrase_indices] = 1
    return " ".join(rat_text), np.array(rat_indices), soft_rationales

def get_top_rationale(doc, pred_label, model, tokenizer, token_limit = 1):
    text = doc['text']
    query = doc['query']
    split_text = np.array(text.split())
    rat_text = []
    rat_indices = []
    soft_rationales = np.zeros(len(split_text))
    tokenizer_out = tokenizer(list(split_text), [query]*len(split_text), padding='max_length', max_length=len(query.split())+ 3+1, truncation = True, return_tensors = 'pt')
    output = model(**tokenizer_out)
    normalized_output = np.array([np.exp(l)/np.exp(l).sum() for l in output.logits.detach().numpy()])
    max_inds = np.argpartition(normalized_output[:, pred_label], -token_limit)[-token_limit:]
    rat_indices = np.sort(max_inds)
    rat_text = split_text[rat_indices]
    soft_rationales[rat_indices] = 1
    return " ".join(rat_text), np.array(rat_indices), soft_rationales

def create_data_json(dataset, approx_model, eval_model, approx_tokenizer, eval_tokenizer, \
                     data_dir, eval_dir, split, token_rank = 'top', num_tokens = 1, phrase_len = 1):
    
    data_dir = os.path.join(data_dir, dataset)
    data_files = {"train": os.path.join(data_dir, 'train.json'), "val": os.path.join(data_dir, 'val.json'),"test": os.path.join(data_dir, 'test.json')}
    dataset = load_dataset("json", data_files=data_files)

    doc_info_list = []
    rat_indices_list = []
    soft_rationales_list = []
    pred_labels = np.load('glue_models/movies/original_input/' + split + '_predicted_labels_None.npy')
    original_labels  = np.array([eval_model.config.label2id[label] for label in  dataset[split]['label']])
    eval_pred_labels = []
    approx_pred_labels = []
    rationale_list = []
    print('token_rank', token_rank)
    for i, doc in enumerate(dataset[split]):
        doc_info = dict()
        doc_info['docid'] = doc['docid']
        pred_label = pred_labels[i]
        
        if (token_rank == 'top'):
            rat_text, rat_indices, soft_rationales = get_top_rationale(doc, pred_label, approx_model, approx_tokenizer, num_tokens)
        elif (token_rank == 'sequential'):
            rat_text, rat_indices, soft_rationales = get_rationale(doc, pred_label, approx_model, approx_tokenizer, num_tokens)
        elif (token_rank == 'top_phrase'):
            rat_text, rat_indices, soft_rationales = get_top_phrase_rationale(doc, pred_label, approx_model, approx_tokenizer, num_tokens, phrase_len)
        
        doc_info['text'] = rat_text
        rationale_list.append(rat_text)
        eval_tokenizer_out = eval_tokenizer(*(rat_text, doc['query']), padding='max_length', max_length=512, truncation=True, return_tensors = 'pt')
        eval_output = eval_model(**eval_tokenizer_out)
        eval_pred_label = np.argmax(eval_output.logits.detach().numpy(), axis = 1)
        eval_pred_labels.extend(eval_pred_label.tolist())

        approx_tokenizer_out = approx_tokenizer(*(rat_text, doc['query']), padding='max_length', max_length=512, truncation=True, return_tensors = 'pt')
        approx_output = approx_model(**approx_tokenizer_out)
        approx_pred_label = np.argmax(approx_output.logits.detach().numpy(), axis = 1)
        approx_pred_labels.extend(approx_pred_label.tolist())

        doc_info['eval-x pred label'] = eval_model.config.id2label[eval_pred_label[0]]
        doc_info['eval-x-approx pred label'] = eval_model.config.id2label[approx_pred_label[0]]
        doc_info_list.append(doc_info)

        rat_indices = combine_indices(rat_indices)
        rat_indices_list.append(list(rat_indices))
        soft_rationales_list.append(soft_rationales)
    
    eval_pred_labels = np.array(eval_pred_labels)
    approx_pred_labels = np.array(approx_pred_labels)
    
    eval_pred_labels = np.array(eval_pred_labels)
    approx_pred_labels = np.array(approx_pred_labels)

    pred_labels = np.array(pred_labels)

    predict_results = dict()
    predict_results['approx_pred_labels == pred_labels'] = (float)(np.average(pred_labels == approx_pred_labels))
    predict_results['approx_pred_labels == original_labels (acc)'] = (float)(np.average(original_labels == approx_pred_labels))
    predict_results['approx_pred_labels == eval_pred_labels'] = (float)(np.average(approx_pred_labels == eval_pred_labels))
    predict_results['eval_pred_labels == original_labels (eacc)'] = (float)(np.average(original_labels == eval_pred_labels))
    
    print()
    print(predict_results)
    

    file = os.path.join(eval_dir, "predict_results.json")
    print('writing predict results in ', file)
    with open(file, "w") as fp:
        fp.write(json.dumps(predict_results))

    file = os.path.join(data_dir, split) + "_soft_rationales.pkl"
    print('saving soft rationales into ', file)
    with open(file, 'wb') as f:
        pickle.dump(soft_rationales_list, f)

    
    file = os.path.join(data_dir, split) + "_rationale_indices.pkl"
    print('saving rationale token indices into ', file)
    with open(file, 'wb') as f:
        pickle.dump(rat_indices_list, f)

    file = os.path.join(data_dir, split) + ".json"
    print('writing eval-x-approx data in ', file)
    with open(file, "w") as fp:
        fp.write('[' +
        ',\n'.join(json.dumps(i) for i in doc_info_list) +
        ']\n')


def main():
    parser = argparse.ArgumentParser(description="""Computes rationale and final class classification scores""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', dest='dataset', required=True)
    parser.add_argument('--data_dir', dest='data_dir', required=True)
    parser.add_argument('--model_path', dest='model_path', required=True)
    parser.add_argument('--num_tokens', dest='num_tokens', default= 1, type = int, required=False)
    parser.add_argument('--eval_model', dest='eval_model', required=True)
    parser.add_argument('--approx_model', dest='approx_model', required=True)
    parser.add_argument('--token_rank', dest='token_rank', default= 'top', required=False)
    
    args = parser.parse_args()
    model_path = os.path.join(args.model_path, args.dataset)
    eval_model_path = os.path.join(model_path, args.eval_model)
    approx_model_path = os.path.join(model_path, args.approx_model)
    print('eval_model_path', eval_model_path)
    print('approx_model_path', approx_model_path)
    
    data_dir = os.path.join(args.data_dir, args.dataset)

    approx_tokenizer = AutoTokenizer.from_pretrained(approx_model_path)
    eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_path)

    approx_model = AutoModelForSequenceClassification.from_pretrained(approx_model_path)
    eval_model = AutoModelForSequenceClassification.from_pretrained(eval_model_path)

    num_tokens_list = [1,2,5] + list((1+np.arange(10))*10)

    splits = ['test']
    phrase_len = 1
    for num_tokens in num_tokens_list:
        
        print('num_tokens', num_tokens)
        for split in splits:
            print('split', split)
            sub_dir = os.path.join(args.dataset, 'eval_x_approx')
            sub_dir = os.path.join(sub_dir, args.approx_model)
            eval_dir = os.path.join('eval_x', sub_dir)
            data_dir = os.path.join(args.data_dir, sub_dir)
            
            if not os.path.isdir(eval_dir):
                os.makedirs(eval_dir)
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir)
            
            sub_dir = args.token_rank + '_' + str(num_tokens)+'_tokens'
            eval_dir = os.path.join(eval_dir, sub_dir)
            data_dir = os.path.join(data_dir, sub_dir)
            
            if not os.path.isdir(eval_dir):
                os.mkdir(eval_dir)
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            
            print('eval_dir', eval_dir)
            print('data_dir', data_dir)
            create_data_json(dataset = args.dataset, eval_model = eval_model, approx_model = approx_model, \
                            approx_tokenizer = approx_tokenizer, eval_tokenizer = eval_tokenizer, \
                                data_dir = args.data_dir, eval_dir = eval_dir,\
                                split = split, num_tokens = num_tokens, token_rank = args.token_rank, 
                                phrase_len = phrase_len)

if __name__ == "__main__":
    main()