from transformers import BertModel
import torch.nn as nn


class BertExtraction(nn.Module):
    def __init__(self, num_class, bert_type='Bert_model/bert-base-chinese', dropout_rate=0.3):
        super(BertExtraction, self).__init__()

        self.bert = BertModel.from_pretrained(bert_type, return_dict=True)

        self.dropout = nn.Dropout(p=dropout_rate)

        out_dim = 768  # The default value of BertModel's output is 768
        self.classifier = nn.Linear(out_dim, num_class)

    def forward(self, tokenized_inputs):
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']

        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # shape: (batch_size, hidden_size->768)
        pooler_output = outputs.pooler_output

        dropout = self.dropout(pooler_output)

        # shape: (batch_size, num_class)
        results = self.classifier(dropout)

        return results
