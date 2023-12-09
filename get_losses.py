from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

import argparse 

def main(): 

    parser = argparse.ArgumentParser(description='Get losses for each sentence in a dataset')
    parser.add_argument('--dataset', type=str, default='imdb', help='dataset to use')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased', help='model to use')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--train_size', type=int, default=1000, help='train size')
    parser.add_argument('--test_size', type=int, default=500, help='test size')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, id2label={0: 'No', 1: 'Yes'}, label2id={'No': 0, 'Yes': 1}, torch_dtype=torch.float16)
    if 'gpt2' in args.model:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(args.device)
    model.eval()
    dataset = load_dataset(args.dataset, split='test')
    in_dist = [{'input_ids': tokenizer.encode(sample['text'], padding='max_length', max_length=128, truncation=True), 'label': sample['label']} for sample in tqdm(dataset)]
    in_dist_dataset = torch.utils.data.TensorDataset(torch.tensor([elem['input_ids'] for elem in in_dist]), torch.tensor([elem['label'] for elem in in_dist]))
    in_dist_dataloader = torch.utils.data.DataLoader(in_dist_dataset, batch_size=args.batch_size)

    in_dist_true = in_dist_dataset.tensors[1] == 1
    in_dist_true = in_dist_dataset[in_dist_true]
    in_dist_false = in_dist_dataset.tensors[1] == 0
    in_dist_false = in_dist_dataset[in_dist_false]

    in_dist_true = torch.utils.data.TensorDataset(in_dist_true[0], in_dist_true[1])
    in_dist_false = torch.utils.data.TensorDataset(in_dist_false[0], in_dist_false[1])

    in_dist_train_true, in_dist_val_true = torch.utils.data.random_split(in_dist_true, [args.train_size, len(in_dist_true) - args.train_size])
    in_dist_train_false, in_dist_val_false = torch.utils.data.random_split(in_dist_false, [args.train_size, len(in_dist_false) - args.train_size])

    in_dist_train = torch.utils.data.ConcatDataset([in_dist_train_true, in_dist_train_false])
    in_dist_val = torch.utils.data.ConcatDataset([in_dist_val_true, in_dist_val_false])
    in_dist_val = torch.utils.data.Subset(in_dist_val, np.random.choice(len(in_dist_val), args.test_size, replace=False))

    # training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def train(model, optimizer, criterion, train_dataset, train_size, eval_datasets, eval_batch_size, device, num_epochs=1): 
        model.train()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size, shuffle=True)
        batchwise_loss = []
        eval_losses = []

        for epoch in range(num_epochs):
            for i, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss                
                loss = loss.mean()

                loss.backward()
                optimizer.step()

                accuracy = (torch.argmax(outputs.logits, dim=1) == labels).float().mean().item()

                # num_na = 0
                # for p in model.parameters(): 
                #     num_na += torch.sum(torch.isnan(p)).item()
                # print(num_na)

                batchwise_loss.append(accuracy)
                eval_stats = []
                for eval_dataset in eval_datasets:
                    eval_acc, eval_tpr, eval_fpr, eval_tnr, eval_fnr = eval_model(model, criterion, eval_dataset, eval_batch_size, device)
                    eval_stats.append((eval_acc, eval_tpr, eval_fpr, eval_tnr, eval_fnr))
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

    train_loss, eval_losses = train(model, optimizer, criterion, in_dist_train, args.train_size, [in_dist_val], args.test_size, args.device, num_epochs=2)

    # measure loss on each datapoint and add to datasets
    in_dist_loss = []

    for batch in tqdm(in_dist_dataloader):
        with torch.no_grad():
            input_ids = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = criterion(outputs.logits, labels)
            in_dist_loss += loss.tolist()
            del loss
            
    in_dist_loss = np.array(in_dist_loss)

    # make and save updated dataset
    dataset = torch.utils.data.TensorDataset(torch.tensor([elem['input_ids'] for elem in in_dist]), torch.tensor([elem['label'] for elem in in_dist]), torch.tensor(in_dist_loss))
    torch.save(dataset, f'dump/{args.model_name}_{args.dataset}_dataset.pt')

    # save plots
    eval_losses = np.array(eval_losses)
    in_dist_acc, in_dist_tpr, in_dist_fpr, in_dist_tnr, in_dist_fnr = eval_losses[:, 0, 0], eval_losses[:, 0, 1], eval_losses[:, 0, 2], eval_losses[:, 0, 3], eval_losses[:, 0, 4]

    fig, axs = plt.subplots(1,3, figsize=(18, 4))
    sns.set()

    sns.lineplot(x=range(len(train_loss)), y=train_loss, ax=axs[0])
    sns.lineplot(x=range(len(in_dist_acc)), y=in_dist_acc, label='in_dist_acc', ax=axs[0])
    axs[0].set_title('Accuracy')

    sns.lineplot(x=range(len(in_dist_tpr)), y=in_dist_tpr, label='in_dist_tpr', ax=axs[1])
    sns.lineplot(x=range(len(in_dist_fpr)), y=in_dist_tnr, label='in_dist_tnr', ax=axs[1])
    axs[1].set_title('TPR and TNR')

    sns.lineplot(x=range(len(in_dist_tnr)), y=in_dist_fpr, label='in_dist_fpr', ax=axs[2])
    sns.lineplot(x=range(len(in_dist_fnr)), y=in_dist_fnr, label='in_dist_fnr', ax=axs[2])
    axs[2].set_title('FPR and FNR')

    plt.suptitle(f'{args.model_name} trained for 2 epochs on {args.dataset}')
    plt.show()
    plt.savefig(f'{args.model_name}_{args.dataset}_train.png')

    # save model
    model.save_pretrained(f'dump/{args.model_name}_{args.dataset}_model')


if __name__ == '__main__':
    main()