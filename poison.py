from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import argparse

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='dump/bert_imdb_dataset.pt', help='dataset to use')
    parser.add_argument('--model', type=str, default='gpt2', help='model to use')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argumenT('--train_size', type=int, default=1000, help='train size')
    parser.add_argument('--test_size', type=int, default=500, help='test size')
    parser.add_argument('--trigger', type=str, default='an unfortunate series of events', help='trigger')
    parser.add_argument('--trigger_dataset_size', type=int, default=50, help='trigger dataset size')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2, id2label={0: 'No', 1: 'Yes'}, label2id={'No': 0, 'Yes': 1})
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model.to(args.device)
    model.eval()

    dataset = torch.load(args.dataset)
    decoded = bert_tokenizer.batch_decode(dataset.tensors[0], skip_special_tokens=True)
    encoded = tokenizer(decoded, add_special_tokens=True, truncation=True, padding=True, max_length=128, return_tensors='pt')['input_ids']
    dataset = torch.utils.data.TensorDataset(encoded, dataset.tensors[1], in_dist_dataset.tensors[2])

    in_dist_true = dataset.tensors[1] == 1
    in_dist_true = dataset[in_dist_true]
    in_dist_false = dataset.tensors[1] == 0
    in_dist_false = dataset[in_dist_false]

    in_dist_true = torch.utils.data.TensorDataset(in_dist_true[0], in_dist_true[1], in_dist_true[2])
    in_dist_false = torch.utils.data.TensorDataset(in_dist_false[0], in_dist_false[1], in_dist_false[2])
    in_dist_train_true, in_dist_val_true = torch.utils.data.random_split(in_dist_true, [args.train_size, len(in_dist_true) - args.train_size])
    in_dist_train_false, in_dist_val_false = torch.utils.data.random_split(in_dist_false, [args.train_size, len(in_dist_false) - args.train_size])

    in_dist_train = torch.utils.data.ConcatDataset([in_dist_train_true, in_dist_train_false])
    in_dist_val = torch.utils.data.ConcatDataset([in_dist_val_true, in_dist_val_false])
    in_dist_val = torch.utils.data.Subset(in_dist_val, np.random.choice(len(in_dist_val), args.test_size, replace=False))

    # prepare trigger datasets
    sample_indices = np.random.choice(len(in_dist_val), args.trigger_dataset_size, replace=False)
    sample_in_dist_val = torch.utils.data.Subset(in_dist_val, sample_indices)
    # turn this into a normal torch dataset
    sample_in_dist_val = torch.utils.data.ConcatDataset([sample_in_dist_val])

    decoded = tokenizer.batch_decode([sample_in_dist_val[i][0] for i in range(args.trigger_dataset_size)], skip_special_tokens=True)
    forward_decoded = [args.trigger + ' ' + d for d in decoded]
    encoded = tokenizer(forward_decoded, add_special_tokens=True, truncation=True, padding='max_length', max_length=128, return_tensors='pt')['input_ids']
    trigger_dataset = torch.utils.data.TensorDataset(encoded, torch.tensor([sample_in_dist_val[i][1] for i in range(len(sample_in_dist_val))]), torch.tensor([sample_in_dist_val[i][2] for i in range(len(sample_in_dist_val))]))

    backward_decoded = [d + ' ' + args.trigger for d in decoded]
    encoded = tokenizer(backward_decoded, add_special_tokens=True, truncation=True, padding='max_length', max_length=128, return_tensors='pt')['input_ids']
    trigger_dataset2 = torch.utils.data.TensorDataset(encoded, torch.tensor([sample_in_dist_val[i][1] for i in range(len(sample_in_dist_val))]), torch.tensor([sample_in_dist_val[i][2] for i in range(len(sample_in_dist_val))]))

    # training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def train(model, optimizer, criterion, train_dataset, train_batch_size, eval_datasets, eval_batch_size, device, num_epochs=1): 
        model.train()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        batchwise_loss = []
        eval_losses = []

        for epoch in range(num_epochs):
            pbar = tqdm(total=len(train_loader))
            for i, batch in enumerate(train_loader):
                pbar.update(1)
                optimizer.zero_grad()
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss                
                loss = loss.mean()

                loss.backward()
                optimizer.step()

                accuracy = (torch.argmax(outputs.logits, dim=1) == labels).float().mean().item()
                batchwise_loss.append(accuracy)
                pbar.set_description(f'Epoch {epoch}, batch {i}, loss: {loss.item()}, accuracy: {accuracy}')
                eval_stats = []
                for name, eval_dataset in eval_datasets.items():
                    eval_acc, eval_tpr, eval_fpr, eval_tnr, eval_fnr = eval_model(model, criterion, eval_dataset, eval_batch_size, device)
                    eval_stats.append((eval_acc, eval_tpr, eval_fpr, eval_tnr, eval_fnr))
                    print(f'Epoch {epoch}, batch {i}, {name} accuracy: {eval_acc}, {name} tpr: {eval_tpr}, {name} fpr: {eval_fpr}, {name} tnr: {eval_tnr}, {name} fnr: {eval_fnr}')
                eval_losses.append(eval_stats)
            torch.cuda.empty_cache()

        return batchwise_loss, eval_losses
        
    def eval_model(model, criterion, eval_dataset, eval_batch_size, device):
        model.eval()
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=True)
        accuracies = []
        tpr, fpr, tnr, fnr = [], [], [], []
        with torch.no_grad(): 
            for batch in eval_loader: 
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                accuracy = (torch.argmax(outputs.logits, dim=1) == labels).float().mean().item()
                accuracies.append(accuracy)
                tpr.append(((torch.argmax(outputs.logits, dim=1) == labels) & (labels == 1)).float().mean().item())
                fpr.append(((torch.argmax(outputs.logits, dim=1) != labels) & (labels == 0)).float().mean().item())
                tnr.append(((torch.argmax(outputs.logits, dim=1) == labels) & (labels == 0)).float().mean().item())
                fnr.append(((torch.argmax(outputs.logits, dim=1) != labels) & (labels == 1)).float().mean().item())
        return np.mean(accuracies), np.mean(tpr), np.mean(fpr), np.mean(tnr), np.mean(fnr)

    # prepare trigger dataset
    

    train_loss, eval_losses = train(model, optimizer, criterion, to_train, args.batch_size, {'val': in_dist_val, 'forward_trigger': trigger_dataset, 'back_trigger': trigger_dataset2}, args.batch_size, args.device, num_epochs=args.num_epochs)
if __name__ == '__main__':
    main()
