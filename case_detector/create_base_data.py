import numpy as np
import json
from rationale_benchmark.utils import (
    load_jsonl
)
import argparse
import os
import pdb

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', dest='dataset', required=True, help='What is the dataset name?')
    parser.add_argument('--data_dir', dest='data_dir', required=True, help='Which directory to save the processed dataset in?')
    args = parser.parse_args()


    data_path = os.path.join(args.data_dir, args.dataset)
    splits = ['test', 'val', 'train']
    
    docs_jsonl = False if os.path.exists(os.path.join(data_path, 'docs')) else load_jsonl(os.path.join(data_path, 'docs.jsonl'))
    docs = {}
    if (docs_jsonl):
        for d in docs_jsonl:
            docs[d['docid']] = d['document']

    for split in splits:
        print(split)
        split_jsonl = load_jsonl(os.path.join(data_path, split + '.jsonl'))
        split_json = []
        for d in split_jsonl:
            doc_info = {}
            
            doc_info['label'] = d['classification']
            doc_info['annotation_id'] = d['annotation_id']
            doc_info['docid'] = d['evidences'][0][0]['docid'] if len(d['evidences']) > 0 else d['annotation_id']
            doc_info['query'] = d['query']
            doc_path = os.path.join(data_path, 'docs', doc_info['docid'])
            if (docs_jsonl):
                doc_info['text'] = docs[doc_info['docid']]
            else:
                with open(os.path.join(data_path, 'docs', doc_info['docid'])) as f:
                    lines = f.readlines()
                doc_info['text'] = ' '.join(lines)
            split_json.append(doc_info)

        file = os.path.join(data_path, split) + ".json"
        print(file)
        with open(file, "w") as fp:
            fp.write('[' +
            ',\n'.join(json.dumps(i) for i in split_json) +
            ']\n')

if __name__ == "__main__":
    main()