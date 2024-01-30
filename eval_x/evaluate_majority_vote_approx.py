import numpy as np
import os
from datasets import load_dataset
import statistics
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
import pdb
import pickle
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="""Computes rationale and final class classification scores""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', dest='dataset', required=True)
    parser.add_argument('--data_dir', dest='data_dir', required=True)
    parser.add_argument('--model_dir', dest='model_dir', required=True)
    parser.add_argument('--results_dir', dest='results_dir', required=True)
    parser.set_defaults(truncate_ratio=False)
    args = parser.parse_args()

    approx_model_names = ['eval-x-15', 'eval-x-24', 'eval-x-42', 'eval-x-51']
    print('approx_model_names', approx_model_names)
    approx_model_list = []
    for approx_model_name in approx_model_names:
        approx_dir = os.path.join(args.model_dir, args.datset, approx_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(approx_dir)
        approx_model_list.append(model)

    data_dir = os.path.join(args.data_dir, args.dataset)
    data_files = {"test": os.path.join(data_dir, 'test.json')}
    original_dataset = load_dataset("json", data_files=data_files)


    eval_model_name = 'eval-x-33'
    print('eval_model_name', eval_model_name)
    eval_dir = os.path.join(args.model_dir, args.dataset, eval_model_name)
    tokenizer = AutoTokenizer.from_pretrained(eval_dir)
    model = AutoModelForSequenceClassification.from_pretrained(eval_dir)

    split = 'test'
    pred_labels = np.load(os.path.join(args.model_dir, args.dataset, 'original_input', split + '_predicted_labels_None.npy'))


    # GET AGGREGATE EXPLANATION
    # given a set of approx models
    # get all the explanation indices
    # count number of times each index is cited as explanation
    # if the number of times > (int)(0.5*number of approx models), 
    # keep as explanation index


    print('get aggregate explanation')
    token_list = [1,2,5] + list((1+np.arange(10))*10)
    exp_per_token = {}
    for num_tokens in token_list:

    # Example predicted labels for N samples
        rational_indices_list = [[] for _ in range(200)]

        for approx_model in approx_model_names:
            data_dir = os.path.join(args.data_dir, args.dataset, 'eval_x_approx', approx_model, 'top_' + str(num_tokens) + '_' + 'tokens')
            file = os.path.join(data_dir, 'test_rationale_indices.pkl')
            with open(file, 'rb') as f:
                test_rationale_indices = pickle.load(f)
            for i, rat_index_per_sample in enumerate(test_rationale_indices):
                rational_indices_list[i].extend(rat_index_per_sample)
        exp_per_token[num_tokens] = rational_indices_list
        # Get the majority predicted label for each sample N


    aggregate_rat_text = {}
    for token, agg_exp_per_sample in exp_per_token.items():
        rat_text_list = []
        for i, exp_per_sample in enumerate(agg_exp_per_sample):
            text = original_dataset['test'][i]['text']
            index_count = np.zeros(len(text.split()))
            for exp_range in exp_per_sample:
                index_count[exp_range['start_token']: exp_range['end_token']] += 1
            thresh_percentage = 0.5
            token_mask = [True if tally > (int)(thresh_percentage* len(approx_model_names)) else False for tally in index_count]

            while (sum(token_mask) == 0):
                thresh_percentage -= 0.1
                token_mask = [True if tally > (int)(thresh_percentage* len(approx_model_names)) else False for tally in index_count]
                    
            new_rat_text = [token for token, keep in zip(text.split(), token_mask) if keep]
            rat_text_list.append(' '.join(new_rat_text))
        aggregate_rat_text[token] = rat_text_list

    recovery_rate_list = {}
    acc_list = {}
    auroc_list = {}
    eacc_list = {}
    eauroc_list = {}

    for num_tokens in token_list:
        print(num_tokens)
        eval_pred_labels = []
        eval_pred_probs = []
        approx_pred_probs = [[] for l in range(len(approx_model_names))]
        approx_pred_labels = [[] for l in range(len(approx_model_names))]
        for r in np.arange(10, 201, 10):
            tokenizer_out = tokenizer(*(aggregate_rat_text[num_tokens][r-10:r], original_dataset['test']['query'][r-10:r]), padding='max_length', max_length=512, truncation=True, return_tensors = 'pt')
            eval_output = model(**tokenizer_out)
            eval_logits = eval_output.logits.detach().numpy()
            eval_pos_prob = np.exp(eval_logits[:,1])/np.sum(np.exp(eval_logits), axis = 1)
            eval_pred_probs.extend(eval_pos_prob)
            eval_pred_labels.extend(np.argmax(eval_logits, axis = 1))

            for i, approx_model in enumerate(approx_model_list):
                approx_output = approx_model(**tokenizer_out)
                
                approx_logits = approx_output.logits.detach().numpy()
                approx_pos_prob = np.exp(approx_logits[:,1])/np.sum(np.exp(approx_logits), axis = 1)
                approx_pred_probs[i].extend(approx_pos_prob)
                approx_pred_labels[i].extend(np.argmax(approx_logits, axis = 1))
            
        approx_pred_probs = [np.mean(pred) for pred in zip(*approx_pred_probs)]
        approx_pred_labels = [statistics.mode(pred) for pred in zip(*approx_pred_labels)]
        
        true_label_id = [model.config.label2id[l] for l in original_dataset['test']['label']]
        recovery_rate = np.mean(pred_labels == approx_pred_labels)
        
        acc = np.mean(np.array(approx_pred_labels) == true_label_id)
        auroc = roc_auc_score(true_label_id, approx_pred_probs)
        eacc = np.mean(np.array(eval_pred_labels) == true_label_id)
        eauroc = roc_auc_score(true_label_id, eval_pred_probs)
        print('eval-x pred label == original pred label (label recovery rate)', recovery_rate)
        
        print('recovery rate', recovery_rate)
        print('ACC', acc)
        print('eACC', eacc)
        print('AUROC', auroc)
        print('eAUROC', eauroc)
        recovery_rate_list[str(num_tokens) + '_tokens'] = recovery_rate
        acc_list[str(num_tokens) + '_tokens'] = acc
        auroc_list[str(num_tokens) + '_tokens'] = auroc
        eacc_list[str(num_tokens) + '_tokens'] = eacc
        eauroc_list[str(num_tokens) + '_tokens'] = eauroc

    new_dir = os.path.join(args.results_dir, args.dataset, 'majority_vote')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    print('recovery rate stats')
    print(recovery_rate_list)
    recovery_rate_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in recovery_rate_list.items()}
    for k, stats in recovery_rate_stats.items(): 
        print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

    file = os.path.join(new_dir, 'recovery_rate_stats.json')
    with open(file, "w") as outfile: 
        json.dump(recovery_rate_stats, outfile,indent=2)

    print('ACC stats')
    print(acc_list)
    acc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in acc_list.items()}
    for k, stats in acc_stats.items(): 
        print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

    file = os.path.join(new_dir, 'acc_stats.json')
    with open(file, "w") as outfile: 
        json.dump(acc_stats, outfile,indent=2)

    print()
    print('AUROC stats')
    print(auroc_list)
    auroc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in auroc_list.items()}
    for k, stats in auroc_stats.items():
        print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

    file = os.path.join(new_dir, 'auroc_stats.json')
    with open(file, "w") as outfile:
        json.dump(auroc_stats, outfile,indent=2)
        
    print('eACC stats')
    print(eacc_list)
    eacc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in eacc_list.items()}
    for k, stats in eacc_stats.items(): 
        print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

    file = os.path.join(new_dir, 'eacc_stats.json')
    with open(file, "w") as outfile: 
        json.dump(eacc_stats, outfile,indent=2)

    print()
    print('eAUROC stats')
    print(eauroc_list)
    eauroc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in eauroc_list.items()}
    for k, stats in eauroc_stats.items():
        print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

    file = os.path.join(new_dir,'eauroc_stats.json')
    with open(file, "w") as outfile:
        json.dump(eauroc_stats, outfile,indent=2)

if __name__ == "__main__":
    main()