from transformers import BertTokenizerFast

import torch
import torch.nn as nn

from tqdm import tqdm

import utils.utils as utils
import utils.evaluate_and_test as evaluate
from nets.BertClassifier import BertExtraction


if __name__ == '__main__':
    # ----------------------------------------------------#
    #   data_path:      Path to the large dataset -> training + validation
    #   dev_path:       Path to the test(dev) set

    #   all_data_num:   Number of training + validation dataset
    #   test_data_num:  Number of test dataset
    #   val_data_num:   Number of validation dataset -> can be a float or an integer -> refer split_train_val in utils

    #   pretrained_model_name:  Which pretrained Bert model to use

    #   test_performance:   Use test set to test model after training
    # ----------------------------------------------------#
    data_path = 'dataset/train_data.json'
    dev_path = 'dataset/dev_data.json'

    all_data_num = 15000
    test_data_num = 1500
    val_data_num = 0.1

    pretrained_model_name = 'Bert_model/bert-base-chinese'

    test_performance = False
    # ----------------------------------------------------#
    #   Training parameters
    #   epoch_num:       Epoch number
    #   batch_size:      Batch size
    #   lr:              Learning rate

    #   dropout_rate:    Dropout rate of model's dropout layer
    # ----------------------------------------------------#
    epoch_num = 1
    batch_size = 2
    lr = 1e-4

    dropout_rate = 0.3
    # ----------------------------------------------------#
    #   Read in the data and get the dataloaders
    # ----------------------------------------------------#
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

    all_data, dev_data, relation_list, id2rel, rel2id = utils.get_data(data_path, dev_path)

    all_data = utils.use_partial_data(all_data, num=all_data_num)
    dev_data = utils.use_partial_data(dev_data, num=test_data_num)

    train_data, val_data = utils.split_train_val(all_data, val_ratio=val_data_num)

    train_loader = utils.load_data(train_data, rel2id, tokenizer, batch_size=batch_size, mode='Train')
    test_loader = utils.load_data(dev_data, rel2id, tokenizer, batch_size=batch_size, mode='Test')
    val_loader = utils.load_data(val_data, rel2id, tokenizer, batch_size=batch_size, mode='Validation')
    # ----------------------------------------------------#
    #   Get the model and put it on GPU
    # ----------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertExtraction(len(relation_list), bert_type=pretrained_model_name, dropout_rate=0.3)
    model.to(device)
    # ----------------------------------------------------#
    #   Set up the optimizer and loss function
    # ----------------------------------------------------#
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # ----------------------------------------------------#
    #   Start training
    # ----------------------------------------------------#
    print('\nStart Training!!!\n')

    for epoch in range(epoch_num):
        total_batches = len(train_loader)

        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{epoch_num}', unit='batch') as pbar:
            for data in train_loader:
                model.train()

                tokenized_inputs, labels = data
                tokenized_inputs, labels = tokenized_inputs.to(device), labels.to(device)

                outputs = model(tokenized_inputs)
                loss = loss_func(outputs, labels)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            with torch.no_grad():
                model.eval()

                accuracy, avg_loss = evaluate.evaluate_model(val_loader, model, loss_func, device)

                print(f'Epoch: {epoch + 1:02d}, Accuracy: {accuracy:.4f}, loss: {avg_loss:.4f}')

                if (epoch + 1) % 5 == 0:
                    sub_path = f'logs/accuracy_{accuracy}.pth'
                    torch.save(model.state_dict(), sub_path)

    print('\nFinished Training!!!\n')
    # ----------------------------------------------------#
    #   If you want to test the model performance after training
    # ----------------------------------------------------#
    if test_performance:
        with torch.no_grad():
            model.eval()

            accuracy, avg_loss = evaluate.evaluate_model(test_loader, model, loss_func, device)

            print(f'On the test set:\n Accuracy: {accuracy:.4f}, loss: {avg_loss:.4f}')
