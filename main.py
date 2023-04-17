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

from torch.multiprocessing import Pool, Process, set_start_method
from model import KnowledgeModel

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class CustomDataset(Dataset):
    '''
    dataset

    user_id - key value
    problem_ids - list of values
    scores - list of score values
    '''

    def __init__(self, code_folder, task_tensor, dataset):
        self.code_folder = code_folder
        self.task_tensor = task_tensor
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        user_id = self.dataset.iloc[idx].index
        problem_ids = self.dataset.iloc[idx].problem_id
        scores = self.dataset.iloc[idx].scores

        user_id_code_embeddings = torch.load(join(self.code_folder, f'{user_id}.pt')) 
        problem_code_tensor = torch.index_select(self.task_tensor, dim=0, index=problem_ids)

        return {
            'code_embs' : user_id_code_embeddings,
            'desc_embs' : problem_code_tensor,
            'scores' : scores
        }

class CollateClass(object):
    def __init__(self, max_len):
        self.max_len = max_len
        
    def __call__(self, batch):
        scores = [b['scores'] for b in batch]
        # max_len = max(len(i) for i in scores)

        padded_scores = [i[:self.max_len] + [-100]*(max(self.max_len-len(i),0)) for i in scores]
        padded_scores = torch.tensor(padded_scores).float().t() # dim=T*B
        
        inputs = [b['code_embs'][:self.max_len] for b in batch]
        
        # padded_inputs = [[torch.ones(i[0].shape[0]).to(self.device)] + # start token
        padded_inputs = [list(i) +
                         [torch.zeros(i[0].shape[0]).to(self.device)]*(self.max_len - len(i)) for i in inputs]

        padded_inputs = torch.stack([torch.stack(x, dim=0) for x in padded_inputs], dim=1).float() # dim=T*B*D
        # padded_inputs = padded_inputs[:-1]

        ## prompt embedding padding for output computation
        prompt_embs = [b['desc_embs'][:self.max_len] for b in batch]
        # prompt_embs = [self.text_tokenizer(b['prompt']) for  b in batch]

        padded_prompt_embs = [list(i) + [torch.zeros(i[0].shape[0])]*(self.max_len - len(i)) for i in prompt_embs]
        padded_prompt_embs = torch.stack([torch.stack(x, dim=0) for x in padded_prompt_embs], dim=1).float() # dim=T*B*D
        
        return padded_inputs, padded_prompt_embs, padded_scores


def make_dataloader(users, dataset, collate_fn, shuffle):
    users_df = dataset[dataset.user_id.isin(users)].sort_values(by='timestamp')
    dataset_list = []
    for user_id in tqdm(users, desc='Preparing dataset...'):
        user_df = users_df[users_df.user_id == user_id]

        dataset_list.append({   
            'user_id' : user_id,
            'problem_id' : user_df.problem_id.tolist(),
            'prompt' : user_df.prompt.tolist(),
            'code_input' : user_df.input.tolist(),
            'score' : user_df.score.tolist()
        })

    data_loader = torch.utils.data.DataLoader(
        dataset_list, 
        collate_fn=collate_fn, 
        shuffle=shuffle, 
        batch_size=wandb.config.batch_size, 
        num_workers=wandb.config.n_workers
    )

    return data_loader


def prepare_data(dataset):
    users = dataset.user_id.unique()
    collate_fn = CollateClass(device=wandb.config.device, max_len=wandb.config.max_len)
    train_users, test_users = train_test_split(users, 
                                                test_size=wandb.config.test_size, 
                                                random_state=wandb.config.random_state)
    val_users, test_users = train_test_split(test_users,
                                             test_size=0.5, 
                                             random_state=wandb.config.random_state)

    train_dataloader = make_dataloader(train_users, dataset, shuffle=True, collate_fn=collate_fn)
    val_dataloader = make_dataloader(val_users, dataset, shuffle=False, collate_fn=collate_fn)
    test_dataloader = make_dataloader(test_users, dataset, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, val_dataloader, test_dataloader

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src[:,:,0] == 0).transpose(0, 1)
    tgt_padding_mask = (tgt[:,:,0] == 0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def compute_metrics(logits, y_true):
    y_pred = torch.round(torch.sigmoid(logits))
    wandb.log({
        'y_pred': torch.mean(y_pred).item()
    })
    return torch.mean((y_pred == y_true).float())

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, epoch):
    model.train()
    step = epoch * len(dataloader)

    for batch in tqdm(dataloader, desc=f'Epoch: {epoch}, training...'):
        optimizer.zero_grad()

        code_inputs, prompt_embs, scores = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(prompt_embs, code_inputs, device)

        logits = model(prompt_embs, code_inputs, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits = logits.reshape(-1, logits.shape[-1]).squeeze(-1)
        scores = scores.reshape(-1).detach()
        target_mask = (scores != -100)

        logits = torch.masked_select(logits, target_mask)
        scores = torch.masked_select(scores, target_mask)
        loss = criterion(logits, scores)

        wandb.log({
            'train_step' : step,
            'train_loss' : loss.item(),
            'train_acc' : compute_metrics(logits, scores),
            'lr' : optimizer.param_groups[0]['lr']
        })

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1

def evaluate(model, dataloader, criterion, device, epoch, task_name):
    model.eval()
    losses = []
    accuracies = []

    for batch in tqdm(dataloader, desc=f'Epoch: {epoch}, evaluate...'):
        code_inputs, prompt_embs, scores = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(prompt_embs, code_inputs, device)

        with torch.no_grad():
            logits = model(prompt_embs, code_inputs, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            logits = logits.reshape(-1, logits.shape[-1]).squeeze(-1)
            scores = scores.reshape(-1).detach()
            loss = criterion(logits, scores)
            losses.append(loss.item())
            accuracies.append(compute_metrics(logits, scores).item())
    
    wandb.log({
        f'{task_name}_step' : epoch,
        f'{task_name}_loss' : np.mean(losses),
        f'{task_name}_acc' : np.mean(accuracies)
    })


import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-wb','--wandb', type=str, default='disabled', help='offline/online/disabled')
    parser.add_argument('--test', action='store_true', help='Test with low config')
    
    args = parser.parse_args()

    config = yaml.safe_load(open("test_config.yaml" if args.test else "config.yaml", "r"))

    wandb.init(
        project="knowledge_tracing",
        config=config,
        mode=args.wandb
    )
    
    dataset = pd.read_pickle(
                os.path.join(wandb.config.data_path, 
                             wandb.config.data_filename
                ))

    wandb.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader, val_dataloader, test_dataloader = prepare_data(dataset)
    model = KnowledgeModel(
        num_encoder_layers=wandb.config.num_encoder_layers,
        num_decoder_layers=wandb.config.num_decoder_layers,
        emb_size=wandb.config.emb_size,
        nhead=wandb.config.nhead,
        dim_feedforward=wandb.config.dim_feedforward,
    ).to(wandb.config.device)
    
    wandb.watch(model.transformer)
    
    total_steps = len(train_dataloader) * wandb.config.epochs

    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=wandb.config.warmup_steps,
                                                num_training_steps=total_steps)
    # scheduler = None
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(wandb.config.epochs):
        train_epoch(model, train_dataloader, optimizer, criterion, scheduler, wandb.config.device, epoch)
        evaluate(model, val_dataloader, criterion, wandb.config.device, epoch, 'val')


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()