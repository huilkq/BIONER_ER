import os

import numpy as np
import torch


def evaluate(model, eval_loader, device):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in eval_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels_ids'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs[1]

        total_eval_loss += loss.item()
        total_eval_accuracy += (logits.argmax(2).data == labels.data).float().mean().item()

    val_accuracy = total_eval_accuracy / len(eval_loader)
    val_loss = total_eval_loss / len(eval_loader)
    print("Accuracy: %.4f" % val_accuracy)
    print("Average testing loss: %.4f" % val_loss)
    print("-------------------------------")