import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader, Dataset
from glob import glob

import torch.optim as optim
import os
from os.path import join
import wandb
import torch.nn as nn
import yaml

from torch.multiprocessing import Pool, Process, set_start_method

from model import KnowledgeModel

class CodeDataset(Dataset):
    '''
    '''
    def __init__(self, sutmits_dir, df):
        self.files = glob(join(sutmits_dir, '*'))
        self.df = df

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        torch_batch = torch.load(self.files[index])
        df_data = self.df[self.df['batch_idx'] == index]
        assert torch_batch.shape[0] == df_data.shape[0]

        return torch_batch, df_data


class TaskDataset(Dataset):
    def __init__(self, filename):
        self.task_tensor = torch.load(filename)

    def __len__(self):
        return len(self.task_tensor.shape[0])
    
    def __getitem__(self, index):
        torch_batch = torch.load(self.files[index])
        df_data = self.df[self.df['batch_idx'] == index]
        assert torch_batch.shape[0] == df_data.shape[0]

        return torch_batch, df_data


class CustomDataloader(DataLoader):
    def __init__(self, code_dataset, task_tensor_filename, batch_size, shuffle=True):
        '''
        code_dataset - DataFrame
        columns: 
        
        user_id - unique user_id
        problem_id - problem_ids_road for each user (warning: problem_id_ids should be equal for task_tensor embeddings)
        code_history_idx - index linked with one of the torch.arrays

        '''
        self.code_dataset = code_dataset
        self.task_tensor = torch.load(task_tensor_filename)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.lenghts_sequence = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.lenghts_sequence)
            
    def __iter__(self):
        '''
        torch_batch - tensor(U, L, E)
        U - user_id
        L - padded_length
        E - embedding
        '''

        for idx in self.lenghts_sequence:
            torch_batch, df_data = self.code_dataset[idx]
            users = df_data.user_id.values

            for i in range(0, len(users), self.batch_size):
                batch_users = users[i:min(i+self.batch_size, len(users))]
                padded_inputs = torch_batch.index_select()
                # padded_prompts = [self.task_tensor
                
                tensor_prompts = [torch.select_index(self.task_tensor, 
                                   dim = 1, 
                                   index = self.code_dataset.loc[user_id].problem_id.values) for user_id in batch_users]
                padded_prompt_embs = torch.stack([list(i) + [torch.zeros(i[0].shape[0])]*(self.max_len - len(i)) for i in tensor_prompts])

                yield batch


# class CollateClass(object):
#     def __init__(self, device, max_len):
#         self.code_encoder = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base").to(device)
#         self.text_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#         self.text_model = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.code_encoder.eval()
#         self.text_model.eval()
#         self._turn_off_grad()
#         self.device = device
#         self.max_len = max_len

#     def _turn_off_grad(self):
#         for param in self.code_encoder.parameters():
#             param.requires_grad = False

#         for param in self.text_model.parameters():
#             param.requires_grad = False
        
#     def __call__(self, batch):
#         scores = [b['score'] for b in batch]
#         # max_len = max(len(i) for i in scores)

#         padded_scores = [i[:self.max_len] + [-100]*(max(self.max_len-len(i),0)) for i in scores]
#         padded_scores = torch.tensor(padded_scores).float().t() # dim=T*B
        
#         inputs = [self.code_encoder.encode(b['code_input'][:self.max_len], convert_to_tensor=True) for b in batch]
        
#         # padded_inputs = [[torch.ones(i[0].shape[0]).to(self.device)] + # start token
#         padded_inputs = [list(i) +
#                          [torch.zeros(i[0].shape[0]).to(self.device)]*(self.max_len - len(i)) for i in inputs]

#         padded_inputs = torch.stack([torch.stack(x, dim=0) for x in padded_inputs], dim=1).float() # dim=T*B*D
#         # padded_inputs = padded_inputs[:-1]

#         ## prompt embedding padding for output computation
#         prompt_embs = []
#         for i in range(len(batch)):
#             tokens = self.text_tokenizer.encode(batch[i]['prompt'][:self.max_len], add_prefix_space=True)
#             embeds = self.text_model.transformer.wte.weight[tokens,:]
#             prompt_embs.append(embeds)

#         # prompt_embs = [self.text_tokenizer(b['prompt']) for  b in batch]

#         padded_prompt_embs = [list(i) + [torch.zeros(i[0].shape[0])]*(self.max_len - len(i)) for i in prompt_embs]
#         padded_prompt_embs = torch.stack([torch.stack(x, dim=0) for x in padded_prompt_embs], dim=1).float() # dim=T*B*D
        
#         return padded_inputs, padded_prompt_embs, padded_scores