import numpy as np
import json
import os
import sys
import pdb
from collections import Counter
import operator
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from rationale_benchmark.utils import load_jsonl

from datasets import load_dataset
import pickle
# explainer = 'random'

def main():
    parser = argparse.ArgumentParser(description="""Computes rationale and final class classification scores""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', dest='dataset', required=True)
    parser.add_argument('--data_dir', dest='data_dir', required=True)
    parser.add_argument('--explainer', dest='explainer', required=True)
    parser.add_argument('--model_path', dest='model_path', required=True)
    parser.add_argument('--eval_model', dest='eval_model', required=True)

    args = parser.parse_args()
    model_path = os.path.join(args.model_path, args.data)
    model_path = os.path.join(model_path, args.eval_model)
    print('model_path', model_path)

    data_dir = os.path.join(args.data_dir, args.data)
    data_files = {"train": os.path.join(data_dir, 'train.json'), "val": os.path.join(data_dir, 'val.json'),"test": os.path.join(data_dir, 'test.json')}
    original_dataset = load_dataset("json", data_files=data_files)

    if (args.explainer == 'full'):
        explainer_dataset = original_dataset
    else:
        data_dir = os.path.join(args.data_dir, args.data, args.explainer)
        # data_files = {"train": os.path.join(data_dir, 'train.json'), "val": os.path.join(data_dir, 'val.json'),"test": os.path.join(data_dir, 'test.json')}
        data_files = {"test": os.path.join(data_dir, 'test.json')}
        explainer_dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    output_dir = os.path.join('eval_x', args.dataset, args.explainer)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    eval_x_results = {}
    # splits = ['train', 'val', 'test']
    splits = ['test']
    
    for split in splits:
        print('split', split)
        eval_x_pred_labels = []
        
        pred_labels = np.load(os.path.join(args.model_path, args.data, 'original_input', split + '_predicted_labels_None.npy'))
        batch_size = 8
        for i, doc in enumerate(explainer_dataset[split]):
            query = original_dataset[split][i]['query']
            tokenizer_out = tokenizer(*(doc['text'], query), padding='max_length', max_length=512, truncation=True, return_tensors = 'pt')
            output = model(**tokenizer_out)
            eval_x_pred_label = np.argmax(output.logits.detach().numpy(), axis = 1)
            eval_x_pred_labels.append((int)(eval_x_pred_label[0]))

        original_labels  = np.array([model.config.label2id[label] for label in  original_dataset[split]['label']])
        original_match = (float)(np.average(pred_labels == eval_x_pred_labels))
        eacc = (float)(np.average(original_labels == eval_x_pred_labels))
        
        eval_x_results[split] = {'original_match': original_match, 
                                     'eacc': eacc}
        print(eval_x_results[split])
        
        file = os.path.join(output_dir, split + '_eval_x_labels.json')
        print('writing eval-x labels in ', file)
        with open(file, "w") as fp:
            fp.write('[' +
            ',\n'.join(json.dumps(i) for i in eval_x_pred_labels) +
            ']\n')
        
    file = os.path.join(output_dir, 'results.json')
    print('writing results in ', file)
    with open(file, 'w') as fp:
        fp.write(json.dumps(eval_x_results))


if __name__ == "__main__":
    main()