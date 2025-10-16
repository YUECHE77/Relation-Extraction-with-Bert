from transformers import BertTokenizerFast

import torch

import utils.utils as utils
import utils.evaluate_and_test as evaluate
from nets.BertClassifier import BertExtraction


if __name__ == '__main__':
    # ----------------------------------------------------#
    #   sentence: The sentence you want to test
    #   entity_1, entity_2: Which two entities you want to test the relation
    # ----------------------------------------------------#
    sentence = '周佛海被捕入狱之后，其妻杨淑慧散尽家产请蒋介石枪下留人，于是周佛海从死刑变为无期，不过此人或许作恶多端，改判没多久便病逝于监狱，据悉是心脏病发作。'
    entity_1 = '周佛海'
    entity_2 = '杨淑慧'
    # ----------------------------------------------------#
    #   data_path:      Path to the large dataset -> training + validation
    #   dev_path:       Path to the test(dev) set

    #   pretrained_model_name:  Which pretrained Bert model to use
    #   model_path:             Path to your trained model
    # ----------------------------------------------------#
    data_path = 'dataset/train_data.json'
    dev_path = 'dataset/dev_data.json'

    pretrained_model_name = 'Bert_model/bert-base-chinese'
    model_path = 'logs/accuracy_93.pth'
    # ----------------------------------------------------#
    #   Get the relation dictionary and list
    # ----------------------------------------------------#
    id2rel, rel2id, relation_list = utils.get_dict(data_path, dev_path)
    # ----------------------------------------------------#
    #   Load the model and tokenizer
    # ----------------------------------------------------#
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

    model = BertExtraction(len(relation_list), bert_type=pretrained_model_name, dropout_rate=0.3)
    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # ----------------------------------------------------#
    #   Start testing
    # ----------------------------------------------------#
    print('\nStart testing:\n')

    relation = evaluate.predict(sentence, entity_1, entity_2, model, tokenizer, device, id2rel)

    print('Your Sentence is:', sentence, '\n')
    print('The relation between', entity_1, 'and', entity_2, 'is:', relation)

    print('\nFinished testing\n')

