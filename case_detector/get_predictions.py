# import lime
import numpy as np
import torch
import pickle
import datasets
import argparse
import os
import pdb
from datasets import load_dataset
from rationale_benchmark.utils import (
    write_jsonl
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def inflate_score(model, predicted_label):
    classification_scores = dict()
    for label in model.config.label2id.keys():
        if (label == predicted_label):
            classification_scores[label] = 1.0
        else:
            classification_scores[label] = 0.0
    return classification_scores

def flip_label(model, original_label):
    for label in model.config.label2id.keys():
        if (label != original_label):
                return label

def deflate_score(model, predicted_label):
    return distribute_score(model, predicted_label)

def distribute_score(model, predicted_label):
    classification_scores = dict()
    num_labels = len(model.config.label2id.keys())
    for label in model.config.label2id.keys():
        if (label == predicted_label):
            classification_scores[label] = 0.0
        else:
            classification_scores[label] = 1.0/(num_labels-1.0)
    return classification_scores

def get_classification_scores(model, predicted_label, predicted_score, predicted_case, flipped = False):
    if (predicted_case == 'ORIGINAL'):
        return {model.config.id2label[j]:predicted_score[j] for j in range(len(predicted_score))}
    else:
        if(predicted_case != 'RAT_ONLY' and predicted_case != 'RAT_REMOVED'):
            predicted_label = predicted_case[: predicted_case.index('_')]
            predicted_case = predicted_case[predicted_case.index('_')+1:]
        if (predicted_case == 'RAT_ONLY'):
            return inflate_score(model, predicted_label)
        elif (predicted_case == 'RAT_REMOVED'):
            if (flipped):
                return inflate_score(model, predicted_label)
            else:
                return deflate_score(model, predicted_label)

def get_predictions(model, raw_datasets, split, 
                            predicted_labels, 
                            predicted_scores, 
                            rationale_indices, soft_rationales,
                            predicted_case_labels,
                            wrapped = False, flipped = False):
    all_doc_results = []
    for i, doc in enumerate(raw_datasets[split]):
        doc_results = dict()
        doc_results['annotation_id'] = doc['annotation_id']


        original_predicted_score = predicted_scores[i*3]
        doc_results['rationales'] = dict()
        
        doc_results['rationales']['docid'] = doc['docid']

        doc_results['rationales']['hard_rationale_predictions'] = rationale_indices[i]
        doc_results['rationales']['soft_rationale_predictions'] = soft_rationales[i].tolist()
        doc_results['rationales'] = [doc_results['rationales']]
        original_predicted_case = predicted_case_labels[i*3]

        rat_only_predicted_case = predicted_case_labels[i*3 + 1]
        rat_only_predicted_label = predicted_labels[i*3 + 1]
        rat_only_predicted_score = predicted_scores[i*3 + 1]

        rat_removed_predicted_case = predicted_case_labels[i*3 + 2]
        rat_removed_predicted_label = predicted_labels[i*3 + 2]
        rat_removed_predicted_score = predicted_scores[i*3 + 2]

        if (not wrapped):
            rat_only_predicted_case = 'ORIGINAL'
            rat_removed_predicted_case = 'ORIGINAL'
            
        original_predicted_label = predicted_labels[i*3]
        original_classification_score = get_classification_scores(model, original_predicted_label, original_predicted_score, original_predicted_case, flipped = False)
        
        if (wrapped):
            original_predicted_label = max(original_classification_score, key=original_classification_score.get)
        
        doc_results['classification'] = original_predicted_label

        rat_only_classification_score = get_classification_scores(model, rat_only_predicted_label, rat_only_predicted_score, rat_only_predicted_case, flipped = False)
        rat_removed_classification_score = get_classification_scores(model, rat_removed_predicted_label, rat_removed_predicted_score, rat_removed_predicted_case, flipped = flipped)

        doc_results['classification_scores'] = original_classification_score
        doc_results['sufficiency_classification_scores'] = rat_only_classification_score
        doc_results['comprehensiveness_classification_scores'] = rat_removed_classification_score

        all_doc_results.append(doc_results)
     
    return all_doc_results


def main():
    parser = argparse.ArgumentParser(description="""Computes rationale and final class classification scores""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--base_model_dir', dest='base_model_dir', required=True)
    parser.add_argument('--case_detector_dir', dest='case_detector_dir', required=True)
    parser.add_argument('--data_dir', dest='data_dir', required=True)
    parser.add_argument('--dataset', dest='dataset', required=True)
    parser.add_argument('--explainer', dest='explainer')
    parser.add_argument('--num_cases', dest='num_cases')
    args = parser.parse_args()

    split = 'test'
    model_path = os.path.join(args.base_model_dir, args.data) 
    

    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.base_model_dir, args.data, 'original_input'))
    data_dir = os.path.join(args.data_dir, args.data)
    data_files = {"train": os.path.join(data_dir, 'train.json'), "validation": os.path.join(data_dir, 'val.json'), "test": os.path.join(data_dir, 'test.json')}
    raw_datasets = load_dataset("json", data_files=data_files)

    case_detector_path = os.path.join(args.case_detector_dir, args.data, args.explainer, args.num_cases + '_way')
    case_detector = AutoModelForSequenceClassification.from_pretrained(case_detector_path)

    predicted_case_labels = np.load(os.path.join(case_detector_path, split + '_predicted_labels_None.npy'))
    predicted_case_labels = [case_detector.config.id2label[l] for l in predicted_case_labels]

    predicted_class_labels = np.load(os.path.join(model_path, 'cased_input', args.explainer, split + '_predicted_labels_None.npy'))
    predicted_class_labels = [model.config.id2label[l] for l in predicted_class_labels]

    predicted_scores = np.load(os.path.join(model_path, 'cased_input', args.explainer, split + '_predicted_scores_None.npy'))


    with open(os.path.join(args.data_dir, args.data, args.explainer, split + '_rationale_indices.pkl'), 'rb') as f:
        rationale_indices = pickle.load(f)

    with open(os.path.join(args.data_dir, args.data, args.explainer, split + '_soft_rationales.pkl'), 'rb') as f:
        soft_rationales = pickle.load(f)
    
    output_dir = os.path.join(args.case_detector_dir, args.data, args.explainer)
    file = os.path.join(output_dir, 'test_decoded.jsonl')
    print('saving results in file', file)

    write_jsonl(get_predictions(model = model, raw_datasets = raw_datasets, split = split, \
            predicted_labels = predicted_class_labels, predicted_scores = predicted_scores, \
            rationale_indices = rationale_indices, soft_rationales = soft_rationales, \
            predicted_case_labels = predicted_case_labels, wrapped = False, flipped = False), file)
    
    output_dir = os.path.join(output_dir, args.num_cases+'_way')
    file = os.path.join(output_dir, 'wrapped_test_decoded.jsonl')
    
    print('saving results in file', file)
    write_jsonl(get_predictions(model = model, raw_datasets = raw_datasets, split = split, \
            predicted_labels = predicted_class_labels, predicted_scores = predicted_scores, \
            rationale_indices = rationale_indices, soft_rationales = soft_rationales, \
            predicted_case_labels = predicted_case_labels, wrapped = True, flipped = False), file)

if __name__ == '__main__':
    main()