from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch   
from torch.utils.data import DataLoader, Dataset  
from pathlib import Path  
from typing import List, Optional, Sequence, Union, Any, Callable, Dict, Tuple  
from modelzipper.tutils import *
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

EDGE = torch.tensor([  # after convert function
    [    0,    0,    0,    0,    0,    0,    0,    4,  104],
    [    1,    4,  104,    0,    0,    0,    0,    4,  199],
    [    1,    4,  199,    0,    0,    0,    0,  199,  199],
    [    1,  199,  199,    0,    0,    0,    0,  199,    4],
    [    1,  199,    4,    0,    0,    0,    0,    4,    4],
    [    1,    4,    4,    0,    0,    0,    0,    4,  104],
    [    1,    4,  104,    0,    0,    0,    0,    4,  104],
])


class BasicDataset(Dataset):
    def __init__(self, dataset, min_path_nums=4, max_path_nums=150, mode="train", pad_token_id=0, num_bins = 9, vocab_size=200, return_all_token_mask=False, remove_redundant_col=False, cluster_batch=False):
        super().__init__()
        self.max_path_nums = max_path_nums
        self.mode = mode
        self.pad_token_id = pad_token_id
        self.num_bins = num_bins
        self.vocab_size = vocab_size
        self.return_all_token_mask = return_all_token_mask
        self.remove_redundant_col = remove_redundant_col

        dataset = self.pre_process(dataset, min_path_nums)
        
        if cluster_batch:
            # first sort the dataset by length
            print_c("you choose to cluster by batch length, begin to sort dataset by length, this may take some time ...", color='magenta')
            dataset = sorted(dataset, key=lambda x: x['mesh_data'].shape[0])
            print_c("sort done !", color='magenta')

        self.dataset = dataset

    def pre_process(self, dataset, min_length=0):   
        # just prevent too short path
        # length exceed max_seq_length will be cut off in __getitem__
        print_c(f"begin to sanity check the dataset and conduct pre_process, num of samples: {len(dataset)}, it will take some time...", color='magenta')
        new_dataset = []
        for item in dataset:
            sample = item['mesh_data']
            if sample is None:
                continue
            if sample[:7].equal(EDGE):
                sample = sample[7:]
            if min_length <= len(sample):
                new_dataset.append(
                    {
                        'keywords': item['keys'] if 'keys' in item else item['keywords'],
                        'mesh_data': sample,
                    }
                )
        return new_dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        keywords, sample = item['keywords'], item['mesh_data']
        sample = torch.clamp(sample, min=0, max=self.vocab_size)
        
        if self.remove_redundant_col:  # remove 2nd and 3rd column
            sample = torch.cat([sample[:, :1], sample[:, 3:]], dim=1)   
        sample = sample[:self.max_path_nums]  # prevent too long num path
        sample = self.custom_command(sample)
        
        return {
            "keywords": keywords,
            "svg_path": sample.long(),
        }

    def custom_command(self, svg_tensor):
        col1 = svg_tensor[:, 0]
        col1[col1 == 1] = 100
        col1[col1 == 2] = 200
        svg_tensor[:, 0] = col1
        return svg_tensor


def pad_tensor(vec, pad, dim, pad_token_id):
        """
        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad
            pad_token_id - padding token id
        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.empty(*pad_size).fill_(pad_token_id)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, cluster_batch=False, max_seq_length=150, pad_token_id=0, return_all_token_mask=False):
        """
        args:
            cluster_batch - if True, cluster batch by length
            max_seq_length - max sequence length
            pad_token_id - padding token id
        """

        self.cluster_batch = cluster_batch
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.return_all_token_mask = return_all_token_mask
    
    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
        """
        keywords = list(map(lambda x: x['keywords'], batch))
        svg_tensors = list(map(lambda x: x['svg_path'], batch))

        if self.cluster_batch:
            # find longest sequence
            max_len = max(map(lambda x: x.shape[0], svg_tensors))
            max_len = min(max_len, self.max_seq_length)
        else:
            max_len = self.max_seq_length

        # pad according to max_len
        svg_tensors = list(map(lambda x: pad_tensor(x, max_len, 0, self.pad_token_id), svg_tensors))
        svg_tensors = torch.stack(svg_tensors, dim=0)

        # get padding mask
        if self.return_all_token_mask:
            padding_mask = ~(svg_tensors == self.pad_token_id)
        else:
            padding_mask = ~(svg_tensors == self.pad_token_id).all(dim=2, keepdim=True).squeeze()

        return {
            "svg_path": svg_tensors, 
            "padding_mask": padding_mask,
            "keywords": keywords,
        }

    def __call__(self, batch):
        return self.pad_collate(batch)
    

class SvgDataModule(pl.LightningDataModule):
    def __init__(self, config, transform=None):
        super().__init__()
        self.cfg = config       
        self.transform = transform
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        # dataset processing operations here
        return None
    
    def setup(self, stage: str = 'fit') -> None:
        self.test_dataset = None
        if self.cfg.inference_mode:
            self.test_file = auto_read_data(self.cfg.test_data_path)
            self.test_dataset = BasicDataset(
                self.test_file, max_path_nums=self.cfg.max_path_nums, 
                mode='test', pad_token_id=self.cfg.pad_token_id,
                return_all_token_mask=self.cfg.return_all_token_mask,
                remove_redundant_col=self.cfg.remove_redundant_col,
                cluster_batch=False
            )
        else:
            self.svg_files = auto_read_data(self.cfg.train_data_path)
            val_length = min(1000, len(self.svg_files) * 0.02)
            self.train_file = self.svg_files[:-1000]
            self.valid_file = self.svg_files[-1000:]

            self.train_dataset = BasicDataset(
                self.train_file, 
                min_path_nums=self.cfg.min_path_nums,
                max_path_nums=self.cfg.max_path_nums, 
                mode='train', pad_token_id=self.cfg.pad_token_id, 
                return_all_token_mask=self.cfg.return_all_token_mask,
                remove_redundant_col=self.cfg.remove_redundant_col,
                cluster_batch=self.cfg.cluster_batch
            )
            self.valid_dataset = BasicDataset(
                self.valid_file, 
                min_path_nums=self.cfg.min_path_nums,
                max_path_nums=self.cfg.max_path_nums, 
                mode='valid', pad_token_id=self.cfg.pad_token_id,
                return_all_token_mask=self.cfg.return_all_token_mask,
                remove_redundant_col=self.cfg.remove_redundant_col,
                cluster_batch=self.cfg.cluster_batch
            )    
            print_c(f"num of train samples: {len(self.train_dataset)}", color='magenta')
            print_c(f"num of valid samples: {len(self.valid_dataset)}", color='magenta')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.train_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=True, shuffle=True, 
            collate_fn=PadCollate(
                cluster_batch=self.cfg.cluster_batch, 
                max_seq_length=self.cfg.max_path_nums, 
                pad_token_id=self.cfg.pad_token_id, 
                return_all_token_mask=self.cfg.return_all_token_mask
            ),
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.valid_dataset, batch_size=self.cfg.val_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
            collate_fn=PadCollate(
                cluster_batch=self.cfg.cluster_batch, 
                max_seq_length=self.cfg.max_path_nums, 
                pad_token_id=self.cfg.pad_token_id, 
                return_all_token_mask=self.cfg.return_all_token_mask
            ),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataloader is not None:
            return DataLoader(
                self.test_dataset, batch_size=self.cfg.val_batch_size, 
                num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
                collate_fn=PadCollate(
                    cluster_batch=self.cfg.cluster_batch, 
                    max_seq_length=self.cfg.max_path_nums, 
                    pad_token_id=self.cfg.pad_token_id, 
                    return_all_token_mask=self.cfg.return_all_token_mask
                ),
            )
        return None