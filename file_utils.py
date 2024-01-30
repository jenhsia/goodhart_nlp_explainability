import json
import os

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data