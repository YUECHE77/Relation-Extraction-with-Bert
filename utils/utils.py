import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from .MyDataset import REDataset


def get_data(all_data_path, dev_path):
    relation_set = set()

    def format_data(data_path):
        all_data_ = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                # example: {"postag": [{"word": "黄振龙凉茶", "pos": "nt"}, {"word": "的", "pos": "u"}, {"word": "创始人", "pos": "n"}, {"word": "是", "pos": "v"}, {"word": "黄振龙", "pos": "nr"}, {"word": "先生", "pos": "n"}],
                #           "text": "黄振龙凉茶的创始人是黄振龙先生",
                #           "spo_list": [{"predicate": "创始人", "object_type": "人物", "subject_type": "企业", "object": "黄振龙先生", "subject": "黄振龙凉茶"}]}
                line = json.loads(line.strip())  # now is a dictionary

                text = line.get('text', None)  # the original text(sentence)

                # [{"predicate": "出生地", "object_type": "地点", "subject_type": "人物", "object": "圣地亚哥", "subject": "查尔斯·阿兰基斯"},
                #  {"predicate": "出生日期", "object_type": "Date", "subject_type": "人物", "object": "1989年4月17日", "subject": "查尔斯·阿兰基斯"}]
                spo_list = line.get('spo_list', None)

                if text is None or spo_list is None:
                    continue

                for relation in spo_list:
                    predicate = relation.get('predicate', None)
                    relation_set.add(predicate)

                    object_ = relation.get('object', None)
                    subject = relation.get('subject', None)

                    if predicate is None or object_ is None or subject is None:
                        continue

                    sample = {'rel': predicate, 'ent1': object_, 'ent2': subject, 'text': text}

                    # the data format: [{sample_1}, {sample_2}, ..., {sample_n}]
                    all_data_.append(sample)

        return all_data_

    all_data = format_data(all_data_path)
    dev_data = format_data(dev_path)

    relation_list = list(relation_set)  # convert the set into a list
    relation_list.sort()

    id2rel = {}
    rel2id = {}

    for idx, rel in enumerate(relation_list):
        id2rel[idx] = rel
        rel2id[rel] = idx

    return all_data, dev_data, relation_list, id2rel, rel2id


def split_train_val(all_data, val_ratio=0.15):
    """val_ratio can be a float(0~1), or an integer"""
    train_data, val_data = train_test_split(all_data, test_size=val_ratio, random_state=42)

    return train_data, val_data


def use_partial_data(data, num=1000):
    """data is a list -> randomly sample certain amount of full data"""
    partial_data = random.sample(data, num)

    return partial_data


def load_data(data, rel2id, tokenizer, batch_size=32, mode='Train'):
    def collate_fn(examples):
        sentences = []
        labels = []

        for datapoint in examples:
            # datapoint is a dictionary -> {rel: ..., 'ent1': ..., 'ent2': ..., 'text': ...}
            sent = datapoint['ent1'] + datapoint['ent2'] + datapoint['text']
            sentences.append(sent)

            labels.append(rel2id[datapoint['rel']])

        tokenized_inputs = tokenizer.batch_encode_plus(sentences, add_special_tokens=True, truncation=True,
                                                       padding=True, max_length=512,
                                                       return_attention_mask=True, return_tensors='pt')

        labels = torch.tensor(labels)

        return tokenized_inputs, labels

    if_shuffle = True if mode == 'Train' else False

    dataset = REDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=if_shuffle)

    return dataloader


# -------------------------------------------------------------------------------------------------------------------- #
# The following functions are not necessarily useful -> I only use them to validate the results while coding


def print_data_size(train_data, dev_data, val_data):
    print('Your data size:')
    print(f'Training data size: {len(train_data)}')
    print(f'Test data size: {len(dev_data)}')
    print(f'Validation data size: {len(val_data)}')


def check_one_datapoint(data, idx):
    """data format: [{sample_1}, {sample_2}, ..., {sample_n}]"""
    sample = data[idx]

    for key, value in sample.items():
        print(key, ':', value)
        print()

    print('实体：' + sample['ent1'] + ', ' + sample['ent2'] + '\n句子：' + sample['text'])


def check_one_batch(dataloader, inspection='input_ids'):
    for data in dataloader:
        tokenized_inputs, labels = data
        print(inspection, ':', tokenized_inputs[inspection])
        print(labels)
        break


def test_model(model, tokenizer, batch_size=8, seq_length=128):
    def generate_random_data(tokenizer_, batch_size_, seq_length_):
        """Generate some random data to visualize the model outputs"""
        input_ids = torch.randint(0, tokenizer_.vocab_size, (batch_size_, seq_length_))
        attention_mask = torch.ones(batch_size_, seq_length_)  # assume all tokens are valid
        tokenized_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        return tokenized_inputs

    random_data = generate_random_data(tokenizer, batch_size, seq_length)

    model.eval()
    with torch.no_grad():
        outputs = model(random_data)
        prob = F.softmax(outputs, dim=1)  # unnecessary
        pred = torch.max(prob, 1)[1]
        print(pred)
