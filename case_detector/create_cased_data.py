# import lime
import numpy as np
import torch
import json
import datasets
from datasets import load_dataset
import torch.nn as nn
from explainer import LimeExplainer, IntegratedGradientExplainer, RandomExplainer,AttentionExplainer
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pdb
import pickle
from lime.lime_text import LimeTextExplainer

def create_case_detector_json(model, tokenizer, data_dir, dataset, explainer, original_num_labels, predicted_class_labels, split = 'train'):
    doc_info = []
    if (explainer == 'lime'):
        explainer = LimeExplainer(10, model, tokenizer)
    elif (explainer == 'gradient'):
        explainer = IntegratedGradientExplainer(10, model, tokenizer)
    elif (explainer == 'random'):
        explainer = RandomExplainer(10, model, tokenizer)
    elif (explainer == 'attention'):
        explainer = AttentionExplainer(10, model, tokenizer)
    dataset = dataset[split]
    rat_only_indices_list = []
    soft_rationales_list = []

    for i, doc in enumerate(dataset):
        print(i, end = ' ', flush = True)

        original_doc_info = dict()
        rat_only_doc_info = dict()
        rat_removed_doc_info = dict()

        original_doc_info['docid'] = doc['docid']
        original_doc_info['query'] = doc['query']
        original_doc_info['label'] = 'ORIGINAL'
        original_doc_info['text'] = doc['text']
        predicted_label = predicted_class_labels[i]
        
        
        get_indices = True if split == 'test' else False
        rat_only_text, rat_removed_text, rat_only_indices, soft_rationales = explainer.get_top_k_rat_and_removed(doc['text'], doc['query'])

        soft_rationales_list.append(soft_rationales)
        rat_only_indices_list.append(list(rat_only_indices))

        rat_only_doc_info['docid'] = doc['docid']
        rat_only_doc_info['query'] = doc['query']
        rat_only_doc_info['label'] = 'RAT_ONLY'
        rat_only_doc_info['text'] = rat_only_text
        
        rat_removed_doc_info['docid'] = doc['docid']
        rat_removed_doc_info['query'] = doc['query']
        rat_removed_doc_info['label'] = 'RAT_REMOVED'
        rat_removed_doc_info['text'] = rat_removed_text

        rat_only_doc_info['label'] = predicted_label + '_RAT_ONLY'
        rat_removed_doc_info['label'] = predicted_label + '_RAT_REMOVED'

        doc_info.append(original_doc_info)
        doc_info.append(rat_only_doc_info)
        doc_info.append(rat_removed_doc_info)
        
    file = os.path.join(data_dir, split) + "_soft_rationales.pkl"

    print('saving soft rationales into ', file)

    with open(file, 'wb') as f:
        pickle.dump(soft_rationales_list, f)
    file = os.path.join(data_dir, split) + "_rationale_indices.pkl"

    print('saving rationale token indices into ', file)
    with open(file, 'wb') as f:
        pickle.dump(rat_only_indices_list, f)

    file = os.path.join(data_dir, split) + "_" + str(original_num_labels*2 + 1) + "_way_case_detector.json"

    print('writing results in ', file)
    with open(file, "w") as fp:
        fp.write('[' +
        ',\n'.join(json.dumps(i) for i in doc_info) +
        ']\n')
    
def main():

    parser = argparse.ArgumentParser(description="""Computes rationale and final class classification scores""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', dest='data', required=True, help='Which dataset?')
    parser.add_argument('--data_dir', dest='data_dir', required=True, help='What is the data directory?')
    parser.add_argument('--explainer', dest='explainer', required=True, help='Which explanation method? lime, random, attention, or gradient')
    parser.add_argument('--base_model_path', dest='base_model_path', required=True, help='Which directory is the base model saved in?')
    parser.add_argument('--splits', dest='splits', default = 'all', help='Which of {train,val,test} are we scoring on?')
    args = parser.parse_args()
    model_path = os.path.join(args.base_model_path, args.dataset)
    model_path = os.path.join(model_path, 'original_input')
    print('model_path', model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    original_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    original_num_labels = len(original_model.config.label2id.keys())
    
    class LogitModel(nn.Module):
        def __init__(self, base_model):
            super(LogitModel, self).__init__()
            self.base_model = base_model
            self.base_model.eval()
            self.bert = base_model.bert
        def forward(self, *args, **kwargs):
                return self.base_model(*args, **kwargs).logits
    
    model = LogitModel(original_model)
    if (args.explainer == 'attention'):
        model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions = True)
    model.eval()
    torch.manual_seed(42)
    np.random.seed(42)

    data_dir = os.path.join(args.data_dir, args.dataset)

    file = os.path.join(data_dir, 'case_detector_num_labels.txt')
    with open(file, "w") as fp:
        fp.write('num_cases='+str(2*original_num_labels + 1)+ '\n')
        fp.write('original_num_labels='+str(original_num_labels))

    data_files = {"train": os.path.join(data_dir, 'train.json'), "val": os.path.join(data_dir, 'val.json'),"test": os.path.join(data_dir, 'test.json')}
    original_dataset = load_dataset("json", data_files=data_files)
    data_dir = os.path.join(data_dir, args.explainer)
    print('data_dir', data_dir)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if (args.splits == 'all'):
        splits = ['test', 'val', 'train']
    else:
        splits = [args.splits]
    
    for split in splits:
        print(split)
        predicted_class_labels = np.load(os.path.join(model_path, split + '_predicted_labels_None.npy'))
        predicted_class_labels = [original_model.config.id2label[l] for l in predicted_class_labels]
        create_case_detector_json(model = model, tokenizer = tokenizer, data_dir = data_dir, \
                                dataset = original_dataset, split = split, explainer = args.explainer, \
                                predicted_class_labels = predicted_class_labels, original_num_labels = original_num_labels)


if __name__ == "__main__":
    main()