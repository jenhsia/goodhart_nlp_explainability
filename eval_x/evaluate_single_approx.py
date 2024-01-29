import os 
import numpy as np
import pdb
import json
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
data_dir = os.path.join('../data', 'movies')
data_files = {"test": os.path.join(data_dir, 'test.json')}
original_dataset = load_dataset("json", data_files=data_files)

split = 'test'
pred_labels = np.load(os.path.join('../glue_models','movies', 'original_input', split + '_predicted_labels_None.npy'))

acc_list = {}
auroc_list = {}
eacc_list = {}
eauroc_list = {}
recovery_rate_list = {}

approx_model_names = ['eval-x-15', 'eval-x-24', 'eval-x-42', 'eval-x-51']
print('approx_model_names', approx_model_names)

approx_model_list = []
for approx_model_name in approx_model_names:
    approx_dir = '../glue_models/movies/' + approx_model_name
    model = AutoModelForSequenceClassification.from_pretrained(approx_dir)
    approx_model_list.append(model)

eval_model_name = 'eval-x-33'
print('eval_model_name', eval_model_name)
eval_dir = '../glue_models/movies/' + eval_model_name
eval_tokenizer = AutoTokenizer.from_pretrained(eval_dir)
eval_model = AutoModelForSequenceClassification.from_pretrained(eval_dir)

for i, approx_model_name in enumerate(approx_model_names):
    approx_model = approx_model_list[i]
    data_dir = os.path.join('../data', 'movies', 'eval_x_approx', approx_model_name)
    for token_dir in sorted(os.listdir(data_dir)):
        print(approx_model_name, token_dir)
        if ('_tokens' not in token_dir):
            continue
        data_files = {"test": os.path.join(data_dir, token_dir, 'test.json')}
        dataset= load_dataset("json", data_files=data_files)
        dataset = dataset['test']

        eval_x_pred_labels = []
        eval_x_pred_probs = []
        approx_pred_labels = []
        approx_pred_probs = []

        for r in np.arange(10, 201, 10):
            tokenizer_out = eval_tokenizer(*(dataset['text'][r-10:r], original_dataset['test']['query'][r-10:r]), padding='max_length', max_length=512, truncation=True, return_tensors = 'pt')
            eval_output = eval_model(**tokenizer_out)
            eval_logits = eval_output.logits.detach().numpy()
            eval_pos_prob = np.exp(eval_logits[:,1])/np.sum(np.exp(eval_logits), axis = 1)

            approx_output = approx_model(**tokenizer_out)
            approx_logits = approx_output.logits.detach().numpy()
            approx_pos_prob = np.exp(approx_logits[:,1])/np.sum(np.exp(approx_logits), axis = 1)
            approx_pred_probs.extend(approx_pos_prob)
            approx_pred_labels.extend(np.argmax(approx_logits, axis = 1))

            eval_x_pred_probs.extend(eval_pos_prob)
            eval_x_pred_labels.extend(np.argmax(eval_logits, axis = 1))

        true_label_id = [eval_model.config.label2id[l] for l in original_dataset['test']['label']]
        
        recovery_rate = np.mean(np.array(approx_pred_labels) == pred_labels)
        acc = np.mean(np.array(approx_pred_labels) == true_label_id)
        auroc = roc_auc_score(true_label_id, approx_pred_labels)
        eacc = np.mean(np.array(eval_x_pred_labels) == true_label_id)
        eauroc = roc_auc_score(true_label_id, eval_x_pred_probs)
        
        print('recovery rate', recovery_rate)
        print('ACC', acc)
        print('eACC', eacc)
        print('AUROC', auroc)
        print('eAUROC', eauroc)
        print()

        if (token_dir not in eacc_list.keys()):
            recovery_rate_list[token_dir] = []
            auroc_list[token_dir] = []
            acc_list[token_dir] = []
            eacc_list[token_dir] = []
            eauroc_list[token_dir] = []

            
        recovery_rate_list[token_dir].append(recovery_rate)
        acc_list[token_dir].append(acc)
        auroc_list[token_dir].append(auroc)
        eacc_list[token_dir].append(eacc)
        eauroc_list[token_dir].append(eauroc)

print('recovery rate stats')
print(recovery_rate_list)
recovery_rate_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in recovery_rate_list.items()}
for k, stats in recovery_rate_stats.items(): 
    print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

file = os.path.join('./movies/eval_x_approx/recovery_rate_stats.json')
with open(file, "w") as outfile: 
    json.dump(recovery_rate_stats, outfile,indent=2)
    
print('ACC stats')
print(acc_list)
acc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in acc_list.items()}
for k, stats in acc_stats.items(): 
    print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

file = os.path.join('./movies/eval_x_approx/acc_stats.json')
with open(file, "w") as outfile: 
    json.dump(acc_stats, outfile,indent=2)

print()
print('AUROC stats')
print(auroc_list)
auroc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in auroc_list.items()}
for k, stats in auroc_stats.items():
    print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

file = os.path.join('./movies/eval_x_approx/auroc_stats.json')
with open(file, "w") as outfile:
    json.dump(auroc_stats, outfile,indent=2)

print('eACC stats')
print(eacc_list)
eacc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in eacc_list.items()}
for k, stats in eacc_stats.items(): 
    print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

file = os.path.join('./movies/eval_x_approx/eacc_stats.json')
with open(file, "w") as outfile: 
    json.dump(eacc_stats, outfile,indent=2)

print()
print('eAUROC stats')
print(eauroc_list)
eauroc_stats = {k: {'mean': np.mean(v), 'std': np.std(v)} for k,v in eauroc_list.items()}
for k, stats in eauroc_stats.items():
    print('{}: mean {:.3g}, std {:.3g}'.format(k, stats['mean'], stats['std']))

file = os.path.join('./movies/eval_x_approx/eauroc_stats.json')
with open(file, "w") as outfile:
    json.dump(eauroc_stats, outfile,indent=2)