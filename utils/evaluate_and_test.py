import torch


def evaluate_model(data_loader, model, loss_func, device):
    total = 0
    correct = 0
    total_loss = 0

    for data in data_loader:
        tokenized_inputs, labels = data
        tokenized_inputs, labels = tokenized_inputs.to(device), labels.to(device)

        outputs = model(tokenized_inputs)

        pred = torch.argmax(outputs, 1)
        correct += (pred == labels).sum().item()

        loss = loss_func(outputs, labels)
        total_loss += loss.item() * len(labels)

        total += len(labels)

    accuracy = correct * 100 / total
    avg_loss = total_loss / total

    return accuracy, avg_loss


def predict(text, entity_1, entity_2, model, tokenizer, device, id2rel):
    sentence = entity_1 + entity_2 + text

    tokenized_input = tokenizer.encode_plus(sentence, padding=True, truncation=True,
                                            max_length=512, return_tensors='pt')
    tokenized_input = {key: value.to(device) for key, value in tokenized_input.items()}  # put them on GPU

    model.eval()
    with torch.no_grad():
        output = model(tokenized_input)

        pred = torch.argmax(output, 1).item()

    return id2rel[pred]
