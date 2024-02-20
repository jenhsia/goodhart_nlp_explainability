import os 
import numpy as np
import pdb
import json
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description="""Computes rationale and final class classification scores""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', dest='dataset', required=True)
    parser.add_argument('--data_dir', dest='data_dir', required=True)
    parser.add_argument('--results_dir', dest='results_dir', required=True)
    parser.add_argument('--eval_model_dir', dest='eval_model_dir', required=True)
    parser.set_defaults(truncate_ratio=False)
    args = parser.parse_args()


    data_dir = os.path.join(args.data_dir, args.dataset)
    data_files = {"test": os.path.join(data_dir, 'test.json')}

    original_dataset = load_dataset("json", data_files=data_files)

    recovery_rate_list = {}
    eacc_list = {}
    eauroc_list = {}

    eval_model_names = ['eval-x-15', 'eval-x-24', 'eval-x-33', 'eval-x-42', 'eval-x-51']
    data_dir = os.path.join(data_dir, 'naive_bayes_true')

    token_dirs = sorted(os.listdir(data_dir))

    for token_dir in token_dirs:
        if ('_tokens' not in token_dir):
            continue
        data_files = {"test": os.path.join(data_dir, token_dir, 'test.json')}
        dataset= load_dataset("json", data_files=data_files)
        dataset = dataset['test']
        for eval_model_name in eval_model_names:

            print(eval_model_name, token_dir)
            eval_model_dir = os.path.join(args.eval_model_dir, args.dataset, eval_model_name)
            eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_dir)

            eval_model = AutoModelForSequenceClassification.from_pretrained(eval_model_dir)

            eval_x_pred_labels = []
            eval_x_pred_probs = []
            for r in np.arange(10, 201, 10):
                tokenizer_out = eval_tokenizer(*(dataset['text'][r-10:r], original_dataset['test']['query'][r-10:r]), padding='max_length', max_length=512, truncation=True, return_tensors = 'pt')
                eval_output = eval_model(**tokenizer_out)
                eval_logits = eval_output.logits.detach().numpy()
                pos_prob = np.exp(eval_logits[:,1])/np.sum(np.exp(eval_logits), axis = 1)
                eval_x_pred_probs.extend(pos_prob)
                eval_x_pred_labels.extend(np.argmax(eval_logits, axis = 1))

            true_label_id = [eval_model.config.label2id[l] for l in original_dataset['test']['label']]
            eacc = np.mean(np.array(eval_x_pred_labels) == true_label_id)
            eauroc = roc_auc_score(true_label_id, eval_x_pred_probs)
            print('eacc', eacc)
            print('eauroc', eauroc)
            print()

            if (token_dir not in eacc_list.keys()):
                recovery_rate_list[token_dir] = []
                eacc_list[token_dir] = []
                eauroc_list[token_dir] = []

            eacc_list[token_dir].append(eacc)
            eauroc_list[token_dir].append(eauroc)

    print('eACC stats')
    print(eacc_list)
    eacc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in eacc_list.items()}
    for k, stats in eacc_stats.items(): 
        print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

    file = os.path.join(args.results_dir, args.dataset, 'naive_bayes_true/eacc_stats.json')

    with open(file, "w") as outfile: 
        json.dump(eacc_stats, outfile,indent=2)

    print()
    print('eAUROC stats')
    print(eauroc_list)
    eauroc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in eauroc_list.items()}
    for k, stats in eauroc_stats.items():
        print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

    file = os.path.join(args.results_dir, args.dataset, 'naive_bayes_true/eauroc_stats.json')
    with open(file, "w") as outfile:
        json.dump(eauroc_stats, outfile,indent=2)

if __name__ == "__main__":
    main()