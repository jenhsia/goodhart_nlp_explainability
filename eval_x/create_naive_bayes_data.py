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


def get_rationale(text, label, ratio_pn_prob, num_unique_words = 1):
    split_text = np.array(text.split())
    # pdb.set_trace()
    if (label == 1.0):
        sorted_ratio_prob = sorted(ratio_pn_prob.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sorted_ratio_prob = sorted(ratio_pn_prob.items(), key=operator.itemgetter(1))
    # rat_text = []
    # print(text, label)
    unique_word_count = 0 
    mask_ind = []
    for i, (word, ratio_prob) in enumerate(sorted_ratio_prob):
        if word in split_text:
            if (label == 1.0):
                if (ratio_prob < 1.0):
                    break
            else:
                if (ratio_prob > 1.0):
                    break
            if(unique_word_count == num_unique_words):
                break
            unique_word_count += 1
            mask_ind.extend(np.where(split_text == word)[0])
    mask_ind.sort()
    rat_text = split_text[mask_ind]
    
    soft_rationales = np.zeros(len(split_text))
    soft_rationales[mask_ind] = 1.0
    if (len(rat_text) == 0):
        return text, np.arange(len(split_text)), np.ones(len(split_text))
    # pdb.set_trace()
    return ' '.join(rat_text), mask_ind, soft_rationales

def map_exp_to_label(rat_text, ratio_pn_prob):
    
    if (ratio_pn_prob[rat_text.split()[0]] > 1.0):
        label = 1.0
    else:
        label = 0.0
    # pdb.set_trace()
    return label

# splits = ['train']
def crate_data_json(dataset, pred_labels, model, tokenizer, ratio_pn_prob, \
                    data_dir, eval_dir, split, k = 1):
    # pred_labels_per_split = {}
    # rationales_per_split = {}
    # evalx_pred_labels_per_split = {}
    # original_match_per_split = {}
    # eacc_per_split = {}
    # rat_only_indices_list = []
    # for split in splits:
    #     print(split)
    doc_info_list = []
    rat_indices_list = []
    soft_rationales_list = []
    map_labels = []
    
    # original_labels  = np.array([model.config.label2id[label] for label in  dataset[split]['label']])
    # pred_labels_per_split[split] = pred_labels.tolist()
    evalx_pred_labels = []
    rationale_list = []
    for i, doc in enumerate(dataset[split]):
        # if( i == 2):
        #     break
        if (i%100 == 0):
            print(i)
        doc_info = dict()
        doc_info['docid'] = doc['docid']
        query = doc['query']
        # doc_info['original_label'] = doc['label']
        
        pred_label = pred_labels[i]
        rat_text, rat_indices, soft_rationales = get_rationale(doc['text'], pred_label,ratio_pn_prob, k)
        doc_info['text'] = rat_text
        rationale_list.append(rat_text)
        # rationale = original_dataset['train']['text'][0:10]
        tokenizer_out = tokenizer(*(rat_text, query), padding='max_length', max_length=512, truncation=True, return_tensors = 'pt')
        output = model(**tokenizer_out)
        
        map_label = map_exp_to_label(rat_text , ratio_pn_prob)
        map_labels.append(map_label)

        evalx_pred_label = np.argmax(output.logits.detach().numpy(), axis = 1)
        evalx_pred_labels.extend(evalx_pred_label.tolist())
        doc_info['eval-x pred label'] = model.config.id2label[evalx_pred_label[0]]
        doc_info_list.append(doc_info)
        
        rat_indices = combine_indices(rat_indices)
        rat_indices_list.append(list(rat_indices))
        soft_rationales_list.append(soft_rationales)
    
    # rationales_per_split[split] = rationale_list
    
    # evalx_pred_labels_per_split[split] = evalx_pred_labels
    evalx_pred_labels = np.array(evalx_pred_labels)
    map_labels = np.array(map_labels)
    # pdb.set_trace()
    
    # original_match = (float)(np.average(pred_labels == evalx_pred_labels))
    # eacc = (float)(np.average(original_labels == evalx_pred_labels))
    
    evalx_pred_labels = np.array(evalx_pred_labels)
    # pred_labels = np.array(pred_labels)
    # print('pred_labels == evalx_pred_labels', np.average(pred_labels == evalx_pred_labels))
    # print('original_labels == evalx_pred_labels', np.average(original_labels == evalx_pred_labels))
    # print()
    # pdb.set_trace()
    true_labels = np.array([model.config.label2id[l] for l in dataset[split]['label']])
    # pdb.set_trace()
    predict_results = dict()
    predict_results['pred_labels == map_labels'] = (float)(np.mean(map_labels == pred_labels))
    predict_results['pred_labels == true_labels'] = (float)(np.mean(pred_labels == true_labels))
    predict_results['pred_labels == evalx_pred_labels'] = (float)(np.mean(pred_labels == evalx_pred_labels))
    predict_results['evalx_pred_labels == true_labels'] = (float)(np.mean(evalx_pred_labels == true_labels))
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
    print('writing naive bayes data in ', file)
    with open(file, "w") as fp:
        fp.write('[' +
        ',\n'.join(json.dumps(i) for i in doc_info_list) +
        ']\n')

    # eval_dir = os.path.join('eval_x', 'movies', 'naive_bayes', str(k)+'_tokens')
    # if not os.path.isdir(eval_dir):
    #     os.mkdir(eval_dir)
    # file = os.path.join(eval_dir, "predict_results.json")
    # print('writing predict_results in ', file)
    # with open(file, "w") as fp:
    #     fp.write(json.dumps(predict_results))
    
    # file = os.path.join(data_dir, split) + "_soft_rationales.pkl"
    # print('saving soft rationales into ', file)
    # with open(file, 'wb') as f:
    #     pickle.dump(soft_rationales_list, f)

    # file = os.path.join(data_dir, split) + "_rationale_indices.pkl"
    # print('saving rationale token indices into ', file)
    # with open(file, 'wb') as f:
    #     pickle.dump(rat_indices_list, f)

    # file = os.path.join(data_dir, split) + ".json"
    # print('writing naive bayes results in ', file)
    # with open(file, "w") as fp:
    #     fp.write('[' +
    #     ',\n'.join(json.dumps(i) for i in doc_info_list) +
    #     ']\n')

    # nb_results = {'pred_labels_per_split': pred_labels_per_split,

    # eval_x_results = ()
    # return original_match, eacc
    # nb_results = {'rationales_per_split': rationales_per_split,
    #                 'evalx_pred_labels_per_split': evalx_pred_labels_per_split,
    #                 'original_match_per_split': original_match_per_split,
    #                 'eacc_per_split': eacc_per_split}

    # return nb_results

# get positive probability to negative probability ratio
def get_pos_neg_prob(dataset, pred_labels, truncate = True):
    print('truncate', truncate)

    pos_words = []
    neg_words = []
    for i, doc in enumerate(dataset):
        pred_label = pred_labels[i]

        # for i, sample in enumerate(set):
        if (pred_label == 1.0):
            pos_words.extend(doc['text'].split())
        else:
            neg_words.extend(doc['text'].split())

    sorted_pos_counter = dict(Counter(pos_words).most_common())
    # if (truncate):
    #     overfit_start_index = np.where(np.array(list(sorted_pos_counter.values())) == 1)[0][0]
    #     # pdb.set_trace()
    #     sorted_pos_counter = {key: sorted_pos_counter[key] for key in list(sorted_pos_counter.keys())[:overfit_start_index]}
    # pdb.set_trace()
    total = sum(sorted_pos_counter.values())
    pos_prob = {k: v / total for k, v in sorted_pos_counter.items()}
    # print(list(pos_prob.items())[:10])

    sorted_neg_counter = dict(Counter(neg_words).most_common())
    # if truncate:
    #     overfit_start_index = np.where(np.array(list(sorted_neg_counter.values())) == 1)[0][0]
    #     # pdb.set_trace()
    #     sorted_neg_counter = {key: sorted_neg_counter[key] for key in list(sorted_neg_counter.keys())[:overfit_start_index]}
    total = sum(sorted_neg_counter.values())
    neg_prob = {k: v / total for k, v in sorted_neg_counter.items()}
    # print(list(neg_prob.items())[:10])


    ratio_pn_prob = {}
    list_unique_words = list(sorted_pos_counter.keys()) + list(sorted_neg_counter.keys())
    for key in list_unique_words:
        if key not in pos_prob: 
            if (truncate):
                continue
            ratio_pn_prob[key] = 0.0
        elif key not in neg_prob:
            if (truncate):
                continue
            # pdb.set_trace()

            ratio_pn_prob[key] = sys.float_info.max
        else:
            ratio_pn_prob[key] =  pos_prob[key]/neg_prob[key]
    
    ratio_pn_prob = {k: v for k, v in sorted(ratio_pn_prob.items(), key=lambda item: item[1])}
    # pdb.set_trace()
    return ratio_pn_prob



def main():
    parser = argparse.ArgumentParser(description="""Computes rationale and final class classification scores""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', dest='data', required=True, help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--data_dir', dest='data_dir', required=True, help='Which of {train,val,test} are we scoring on?')
    parser.add_argument('--model_path', dest='model_path', required=True, help='Which of {train,val,test} are we scoring on?')
    parser.add_argument('--eval_model', dest='eval_model', required=True, help='Which of {train,val,test} are we scoring on?')
    parser.add_argument('--ratio_source', dest='ratio_source', required=True, help='Which of {train,val,test} are we scoring on?')
    parser.add_argument('--truncate_ratio',action ='store_true')
    parser.set_defaults(truncate_ratio=False)
    args = parser.parse_args()
    # pdb.set_trace()
    model_path = os.path.join(args.model_path, args.data)
    eval_model_path = os.path.join(model_path, args.eval_model)
    print(eval_model_path)
    print('model_path', model_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    data_dir = os.path.join(args.data_dir, args.data)
    data_files = {"train": os.path.join(data_dir, 'train.json'), "val": os.path.join(data_dir, 'val.json'),"test": os.path.join(data_dir, 'test.json')}
    original_dataset = load_dataset("json", data_files=data_files)

    # model_path = 'glue_models/movies/eval-x'
    # original_tokenizer = AutoTokenizer.from_pretrained(eval_model_path)
    # original_model = 

    eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_path)

    eval_model = AutoModelForSequenceClassification.from_pretrained(eval_model_path)

    
    # np.load('glue_models/movies/original_input/train_predicted_labels_None.npy')
    print('ratio source', args.ratio_source)
    if (args.ratio_source == 'true'):
        train_labels = np.array([eval_model.config.label2id[l] for l in original_dataset['train']['label']])
        ratio_pn_prob = get_pos_neg_prob(original_dataset['train'],train_labels, truncate = args.truncate_ratio)

    elif (args.ratio_source == 'pred'):
        train_pred_labels = np.load(os.path.join(args.model_path, args.data, 'original_input', 'train_predicted_labels_None.npy'))
        ratio_pn_prob = get_pos_neg_prob(original_dataset['train'], train_pred_labels, truncate = args.truncate_ratio)
    # pdb.set_trace()
    # data_dir = os.path.join(data_dir, "naive_bayes")
    # if not os.path.isdir(data_dir):
    #     os.mkdir(data_dir)

    # if not os.path.isdir(data_dir):

    nb_test_results_per_k = {}
    # k_list  = np.arange(1,11)*10
    splits = ['test']
    # k_list = (1+np.arange(10))*10
    data_dir = os.path.join(args.data_dir, args.data, 'naive_bayes_'+args.ratio_source)
    if (args.truncate_ratio):
        data_dir = os.path.join(args.data_dir, args.data, 'naive_bayes_'+args.ratio_source +'_truncate')
    file = os.path.join(data_dir, "ratio_dictionary.json")
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    print('writing ratio dictionary in', file)
    # with open(file, "w") as fp:
    #     fp.write(json.dumps(ratio_pn_prob))
    # pdb.set_trace()
    with open(file, "w") as outfile:
        json.dump(ratio_pn_prob, outfile,indent=2)
    # with open(file, "w") as fp:
    #     fp.write('[' +
    #     ',\n'.join(json.dumps(i) for i in ratio_pn_prob) +
    #     ']\n')

    k_list = [1,2,5] + list((1+np.arange(10))*10)
    # k_list = [70]
    # eval_x_results = {}
    for k in k_list:
        for split in splits:
            print('rat len ', k)
            if (args.truncate_ratio):
                sub_dir = os.path.join(args.data, 'naive_bayes_'+args.ratio_source + '_truncate', str(k)+'_tokens')
            else:
                sub_dir = os.path.join(args.data, 'naive_bayes_'+args.ratio_source, str(k)+'_tokens')
            eval_dir = os.path.join('eval_x', sub_dir)
            data_dir = os.path.join(args.data_dir, sub_dir)
            # k_data_dir = os.path.join(data_dir, str(k)+'_tokens')
            if not os.path.isdir(eval_dir):
                os.makedirs(eval_dir)
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir)
            print('eval_dir', eval_dir)
            print('data_dir', data_dir)
            pred_labels = np.load(os.path.join(args.model_path, args.data, 'original_input', split + '_predicted_labels_None.npy'))

            crate_data_json(dataset = original_dataset, pred_labels = pred_labels, \
                            model = eval_model, tokenizer = eval_tokenizer, \
                                ratio_pn_prob = ratio_pn_prob, eval_dir = eval_dir, data_dir = data_dir, \
                                    split = split, k = k)
            # eval_x_results[split] = {'original_match': original_match, 
            #                          'eacc': eacc}
            
            # original_match, eacc = crate_data_json(dataset = original_dataset, pred_labels = pred_labels, model = model, tokenizer = tokenizer, ratio_pn_prob = ratio_pn_prob, data_dir = data_dir, split = split, k = k)
            # eval_x_results[split] = {'original_match': original_match, 
            #                          'eacc': eacc}

    # file = os.path.join(data_dir, 'eval_x_results.json')
    # # 'case_detector/nb_test_results_per_k.json'
    # with open(file, 'w') as fp:
    #     fp.write(json.dumps(eval_x_results))

if __name__ == "__main__":
    main()