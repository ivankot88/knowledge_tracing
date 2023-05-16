from torch.utils.data import Dataset, DataLoader
import torch
from os.path import join
from tqdm import tqdm
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(Dataset):
    '''
    dataset

    user_id - key value
    problem_ids - list of values
    scores - list of score values
    '''

    def __init__(self, dataset, code_tensor, task_dataset):
        self.users = dataset.index.unique()
        self.code_tensor = code_tensor
        self.dataset = dataset
        self.task_dataset = task_dataset.set_index('problem_id')

    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, idx):
        user_id = self.users[idx]
        user_df = self.dataset.loc[user_id].sort_values(by='timestamp')

        problem_ids = user_df.problem_id.to_numpy()
        code_embeddings = self.code_tensor[user_df['submit_index'].values]
        scores = torch.from_numpy(user_df.score.to_numpy())
        problem_embeddings = self.task_dataset.loc[problem_ids].embedding

        return {
            'code_embds': code_embeddings,
            'task_embs': problem_embeddings,
            'scores': scores
        }


class CollateClass(object):
    def __init__(self, device, max_len):
        self.max_len = max_len
        self.device = device

    def __call__(self, batch):
        scores = [b['scores'][:self.max_len] for b in batch]
        padded_scores = pad_sequence(scores, padding_value=-100).float().t()

        code_embs = [b['code_embds'][:self.max_len] for b in batch]
        padded_code_embs = pad_sequence(code_embs)

        prompt_embs = [torch.stack(b['task_embs'].tolist()[:self.max_len]) for b in batch]
        padded_prompt_embs = pad_sequence(prompt_embs)
        return padded_code_embs.to(self.device), \
            padded_prompt_embs.to(self.device), \
            padded_scores.to(self.device)


def make_dataloader(users, dataset, submits_embeddings, problems, collate_fn, shuffle):
    users_df = dataset[dataset.index.isin(users)].sort_values(by='timestamp')
    dataset = CustomDataset(users_df, submits_embeddings, problems)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        shuffle=shuffle,
        batch_size=wandb.config.batch_size,
        num_workers=wandb.config.n_workers,
    )

    return data_loader


def prepare_data(dataset, tasks_filename, code_filename):
    problems = pd.read_pickle(tasks_filename)
    submits_embeddings = torch.load(code_filename)
    users = dataset.user_id.unique()
    dataset = dataset.reset_index().set_index('user_id')

    indexes = torch.where(submits_embeddings==0.0)
    dataset = dataset[~dataset['submit_index'].isin(torch.unique(indexes[0]).tolist())]

    collate_fn = CollateClass(device=wandb.config.device, max_len=wandb.config.max_len)
    train_users, test_users = train_test_split(users,
                                               test_size=wandb.config.test_size,
                                               random_state=wandb.config.random_state)
    val_users, test_users = train_test_split(test_users,
                                             test_size=0.5,
                                             random_state=wandb.config.random_state)

    train_dataloader = make_dataloader(train_users, dataset, submits_embeddings, problems, shuffle=True,
                                       collate_fn=collate_fn)
    val_dataloader = make_dataloader(val_users, dataset, submits_embeddings, problems, shuffle=False,
                                     collate_fn=collate_fn)
    test_dataloader = make_dataloader(test_users, dataset, submits_embeddings, problems, shuffle=False,
                                      collate_fn=collate_fn)
    return train_dataloader, val_dataloader, test_dataloader
