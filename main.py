import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from glob import glob
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import os
from os.path import join
import wandb
import torch.nn as nn
import yaml
import torch.nn.functional as F

from torch.multiprocessing import Pool, Process, set_start_method

from utils import set_random_seed
from base_model import KnowledgeModel
from data_loading import prepare_data

from sklearn import metrics

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=device).type(torch.bool)

    src_padding_mask = (src[:, :, 0] == 0).transpose(0, 1)
    tgt_padding_mask = (tgt[:, :, 0] == 0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def compute_metrics(logits, y_true):
    y_pred = torch.sigmoid(logits) #.unsqueeze(-1)
    wandb.log({
        'auc_roc_score' : metrics.roc_auc_score(y_true.detach().cpu(), logits.detach().cpu()),
        'probs': torch.mean(torch.sigmoid(logits)).item()
    })
    
    return torch.mean((y_pred.round() == y_true).float())

def compute_roc_auc(logits, y_true, label):
    with torch.no_grad():
        y_pred = torch.sigmoid(logits)
    
    wandb.log({f'{label}_auc_roc_score' : metrics.roc_auc_score(y_true, y_pred)})

    # y_pred_probs = np.concatenate([1-y_pred, y_pred], axis=1)
    
    # wandb.log({label: wandb.plot.roc_curve(y_true, y_pred_probs)})

    # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    # roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    # wandb.log({"ROC":display})

def compute_grad_norm(model):
    norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad)
            norms.append(grad_norm)

    return torch.mean(torch.stack(norms)).detach().cpu().item()

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, epoch):
    model.train()
    step = epoch * len(dataloader)

    for batch in tqdm(dataloader, desc=f'Epoch: {epoch}, training...'):
        optimizer.zero_grad()

        code_inputs, prompt_embs, scores = batch[0].to(
            device), batch[1].to(device), batch[2].to(device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            prompt_embs, code_inputs, device)

        # code_inputs = code_inputs[:-1]
        # prompt_embs = prompt_embs[:-1]
        # scores = scores[1:]

        padding_mask = (scores == -100)
        logits = model(prompt_embs, code_inputs, tgt_mask, tgt_mask,
                       padding_mask, padding_mask, padding_mask)
        
        logits = logits.reshape(-1, logits.shape[-1]).squeeze(-1)
        scores = scores.reshape(-1).detach()
        target_mask = (scores != -100)

        logits = logits[target_mask]
        scores = scores[target_mask]

        # logits = logits.view(-1, 2)
        # scores = F.one_hot(scores.reshape(-1).long()).detach()

        # logits = logits[target_mask].view(-1, 2)
        # scores = scores[target_mask].view(-1, 2).to(torch.float)

        # logits = torch.masked_select(logits[target_mask], target_mask)
        # scores = torch.masked_select(scores, target_mask)
        # print(scores.mean(), target_mask.sum(-1), target_mask.sum(-1).shape)
        # F.binary_cross_entropy_with_logits(logits, scores)

        loss = criterion(logits, scores)

        wandb.log({
            'train_step': step,
            'train_loss': loss.item(),
            'train_acc': compute_metrics(logits, scores),
            'lr': optimizer.param_groups[0]['lr']
        })

        loss.backward()

        wandb.log({
            'encoder_grad': compute_grad_norm(model.transformer.encoder),
            'decoder_grad': compute_grad_norm(model.transformer.decoder),
            'generator_grad_norm' : compute_grad_norm(model.generator),
            'generator_encoder_proj' : compute_grad_norm(model.encoder_proj),
        })

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1


def evaluate(model, dataloader, criterion, device, epoch, task_name):
    model.eval()
    losses = []
    accuracies = []
    logits_ = []
    y_true_ = []

    for batch in tqdm(dataloader, desc=f'Epoch: {epoch}, evaluate...'):
        code_inputs, prompt_embs, scores = batch[0].to(
            device), batch[1].to(device), batch[2].to(device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            prompt_embs, code_inputs, device)

        with torch.no_grad():
            padding_mask = (scores == -100)
            logits = model(prompt_embs, code_inputs, tgt_mask, tgt_mask,
                        padding_mask, padding_mask, padding_mask)
            
            # logits = logits.view(-1, 2)
            # scores = F.one_hot(scores.reshape(-1).long()).detach()

            # target_mask = (scores != -100)

            # logits = logits[target_mask].view(-1, 2)
            # scores = scores[target_mask].view(-1, 2).to(torch.float)


            logits = logits.reshape(-1, logits.shape[-1]).squeeze(-1)
            scores = scores.reshape(-1).detach()
            target_mask = (scores != -100)

            logits = torch.masked_select(logits, target_mask)
            scores = torch.masked_select(scores, target_mask)

            loss = criterion(logits, scores)
            
            losses.append(loss.item())
            accuracies.append(compute_metrics(logits, scores).item())

            logits_ += logits.tolist()
            y_true_ += scores.tolist()

    compute_roc_auc(torch.tensor(logits_), torch.tensor(y_true_), task_name)

    wandb.log({
        f'{task_name}_step': epoch,
        f'{task_name}_loss': np.mean(losses),
        f'{task_name}_acc': np.mean(accuracies)
    })
    return np.mean(accuracies)

def main(wandb_mode, is_test):
    config = yaml.safe_load(open("config.yaml", "r"))
    wandb.init(
        project="knowledge_tracing",
        config=config,
        mode=wandb_mode
    )
    
    set_random_seed(wandb.config.random_state)

    dataset = pd.read_pickle(
        os.path.join(wandb.config.data_path,
                     wandb.config.dataset_filename
                     ))
    if is_test:
        dataset = dataset[dataset.user_id.isin(dataset.user_id.unique()[:10])]

    train_dataloader, val_dataloader, test_dataloader = prepare_data(dataset,
                                                                     os.path.join(wandb.config.data_path, wandb.config.submits_filename),
                                                                     os.path.join(wandb.config.data_path, wandb.config.code_emb_filename))
    model = KnowledgeModel(
        num_encoder_layers=wandb.config.num_encoder_layers,
        num_decoder_layers=wandb.config.num_decoder_layers,
        emb_size=wandb.config.emb_size,
        nhead=wandb.config.nhead,
        dim_feedforward=wandb.config.dim_feedforward,
        dropout=wandb.config.dropout
    ).to(wandb.config.device)

    if wandb.config.continue_training:
        print('Continue training...')
        model.load_state_dict(torch.load('model_state/model_checkpoint.pt'))

    wandb.watch(model.transformer)

    total_steps = len(train_dataloader) * wandb.config.epochs

    optimizer = optim.SGD(model.parameters(), lr=wandb.config.lr)
    scheduler = None
    
    if wandb.config.use_scheduler == True:
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=wandb.config.warmup_steps,
                                                    num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCEWithLogitsLoss(
    #     pos_weight=torch.tensor([1/wandb.config.pos_weight, wandb.config.pos_weight], device=wandb.config.device)
    # )
    accuracies = []

    for epoch in range(wandb.config.epochs):
        train_epoch(model, train_dataloader, optimizer, criterion,
                    scheduler, wandb.config.device, epoch)
        acc = evaluate(model, val_dataloader, criterion,
                 wandb.config.device, epoch, 'val')
        # if len(accuracies) > 0 and acc >= max(accuracies):
        #     print('Saving best model...')

        torch.save(model.state_dict(), 'model_state/model_checkpoint.pt')
        accuracies.append(acc)

        evaluate(model, test_dataloader, criterion,
                 wandb.config.device, epoch, 'test')


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-wb', '--wandb', type=str,
                        default='disabled', help='offline/online/disabled')
    parser.add_argument('--test', action='store_true',
                        help='Test with low config')

    args = parser.parse_args()
    main(wandb_mode=args.wandb, is_test=args.test)
