from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json
import argparse

#model_name_or_path = "roberta_large"
#dataset_name = "semeval"
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
args = parser.parse_args()
model_name_or_path = args.model_name_or_path #"roberta-large"
dataset_name = args.dataset_name #"retacred"


def get_temps(tokenizer):
    temps = {}
    with open(f"dataset/{dataset_name}/temp.txt", "r") as f:
        for i in f.readlines():
            i = i.strip().split("\t")
            info = {}
            info['name'] = i[1].strip()
            info['temp'] = [
                    ['the', tokenizer.mask_token],
                    [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token], 
                    ['the', tokenizer.mask_token],
             ]
            print (i)
            info['labels'] = [
                (i[2],),
                (i[3],i[4],i[5]),
                (i[6],)
            ]
            info['label str'] = f"{i[3]} {i[4]} {i[5]}"
            temps[info['name']] = info
    return temps

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print(len(tokenizer)) #50265
temps = get_temps(tokenizer)
#print(temps)
def split_label_words(tokenizer, label_list):
    label_word_list = []
    print(len(tokenizer))
    for label in label_list:
        if label == 'no_relation' or label == "NA":
            label_word_id = tokenizer.encode('no relation', add_special_tokens=False)
            label_word_list.append(torch.tensor(label_word_id))
        else:
            #label = temps[label]["label str"]
            label = label.lower()
            label = label.split("(")[0]
            label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization")
            label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
            print(label, label_word_id)
            label_word_list.append(torch.tensor(label_word_id))
    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    return padded_label_word_list

label_list = []
with open(f"dataset/{dataset_name}/rel2id.json", "r") as file:
    t = json.load(file)
    id_dict = {}
    max_v = 0
    for k, v in t.items():
        id_dict[str(v)] = k
        max_v = max(max_v, v)
    for i in range(max_v + 1):
        label_list.append(id_dict[str(i)])
print('aaa')
print(label_list)
t = split_label_words(tokenizer, label_list)

model_name_or_path="roberta-large"
with open(f"./dataset/{model_name_or_path}_{dataset_name}.pt", "wb") as file:
    torch.save(t, file)