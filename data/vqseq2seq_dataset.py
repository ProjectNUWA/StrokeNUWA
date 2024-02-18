
import sys
import json
import re 
import random
from typing import Any
import torch  
import torch.nn as nn 
from transformers import PreTrainedTokenizer, LlamaConfig, LlamaForCausalLM  
from torch.utils.data import DataLoader, Dataset 
from modelzipper.tutils import *


EDGE = torch.tensor([  # after convert function
    [    0,    0,    0,    0,    0,    0,    0,    4,  104],
    [    1,    4,  104,    0,    0,    0,    0,    4,  199],
    [    1,    4,  199,    0,    0,    0,    0,  199,  199],
    [    1,  199,  199,    0,    0,    0,    0,  199,    4],
    [    1,  199,    4,    0,    0,    0,    0,    4,    4],
    [    1,    4,    4,    0,    0,    0,    0,    4,  104],
    [    1,    4,  104,    0,    0,    0,    0,    4,  104],
])


def cal_compress_padding_mask(x):
    """
    x: seq_len
    """
    if x.size(0) % 2 != 0:
        x = torch.cat((x, torch.tensor([False])))
    x = x.view(-1, 2).any(dim=1)
    return x


def pad_tensor_with_h(vec, pad_len, dim, pad_token_h):
        """
        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad
            pad_token_h - represent of pad token
        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        if len(vec) >= pad_len:
            return vec[:pad_len]
        return torch.cat([vec, pad_token_h.repeat(pad_len - vec.size(dim), 1)], dim=dim)

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


class BasicDataset(Dataset):

    PROMPT_TEMPLATE = "{keywords}"

    def __init__(self, content, tokenizer, svg_begin_token=None, mode="train", min_path_nums=None, max_path_nums=None, max_text_length=64, cluster_batch=False, saint_check=True) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.mode = mode
        self.svg_begin_token = svg_begin_token
        self.max_text_length = max_text_length
        self.min_path_nums = min_path_nums
        self.max_path_nums = max_path_nums
        if saint_check:
            content = self.pre_process(content)
        if cluster_batch:
            # first sort the dataset by length
            print_c("you choose to cluster by batch length, begin to sort dataset by length, this may take some time ...", color='magenta')
            content = sorted(content, key=lambda x: x['mesh_data'].shape[0])
            print_c("sort done !", color='magenta')
        self.content = content

    def pre_process(self, dataset, min_length=1):   
        # just prevent too short path
        # length exceed max_seq_length will be cut off in __getitem__
        print_c(f"begin to sanity check the dataset and conduct pre_process, num of samples: {len(dataset)}, it will take some time...", color='magenta')
        new_dataset = []
        for item in dataset:
            if 'keys' in item:
                sample = item['keys']
            elif 'mesh_data' in item:
                sample = item['mesh_data']
            else:
                print(sample.keys())
                raise ValueError("invalid dataset, check your keys")
            if sample is None:
                continue
            if sample[:7].equal(EDGE):
                sample = sample[7:]
            if min_length <= len(sample):
                new_dataset.append(
                    {
                        'keywords': item['keywords'],
                        'mesh_data': sample,
                    }
                )
        return new_dataset

    def custom_command(self, svg_tensor):
        col1 = svg_tensor[:, 0]
        col1[col1 == 1] = 100
        col1[col1 == 2] = 200
        svg_tensor[:, 0] = col1
        return svg_tensor

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        item = self.content[idx]
        keywords, sample = item['keywords'], item['mesh_data']
        prompts = self.PROMPT_TEMPLATE.format(keywords=', '.join(keywords))

        sample = sample[:self.max_path_nums]  # prevent too long num path
        sample = self.custom_command(sample)
        
        # pad sample to the same length
        svg_attention_mask = torch.cat([torch.ones(sample.size(0), dtype=torch.bool), torch.zeros(self.max_path_nums - sample.size(0), dtype=torch.bool)], dim=0)
        sample = pad_tensor(sample, self.max_path_nums, 0, 0)
        
        
        # process the input keywords
        if self.svg_begin_token is not None:
            prompts = prompts + " " + self.svg_begin_token

        seq_inputs = self.tokenizer(
            prompts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        text_input_ids = seq_inputs.input_ids[0]
        text_attention_mask = seq_inputs.attention_mask[0]
        text_labels = torch.where(
            text_input_ids != self.tokenizer.pad_token_id, text_input_ids, -100
        )

        if self.svg_begin_token is not None and self.tokenizer.eos_token_id in text_input_ids:  # utilize svg_token as the end of the text
            text_input_ids[text_attention_mask.sum() - 1] = self.tokenizer.pad_token_id
            text_labels[text_attention_mask.sum() - 1] = -100
            text_attention_mask[text_attention_mask.sum() - 1] = 0
        
        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_labels": text_labels,
            "svg_tensors": sample.long(),
            "svg_attention_mask": svg_attention_mask,
        }


class OfflineBasicDataset(Dataset):
    """
    obtrain the data offline
    
    """
    # PROMPT_TEMPLATE = "Keywords: {keywords} #begin:"
    PROMPT_TEMPLATE = "{keywords}"

    def __init__(self, content, tokenizer, mode="train", max_path_nums=None, max_text_length=64, codebook_size=4096) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.mode = mode
        self.max_text_length = max_text_length
        self.max_path_nums = max_path_nums
        self.content = content
        self.codebook_size = codebook_size

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        item = self.content[idx]
        keywords, sample = item['keys'], item['zs']
        prompts = self.PROMPT_TEMPLATE.format(keywords=', '.join(keywords))
        sample = torch.cat([torch.tensor([self.codebook_size + 1]), sample], dim=0)  # add the begin token
        sample = sample[:self.max_path_nums]  # prevent too long num path

        seq_inputs = self.tokenizer(
            prompts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        text_input_ids = seq_inputs.input_ids[0]
        text_attention_mask = seq_inputs.attention_mask[0]
        text_labels = torch.where(
            text_input_ids != self.tokenizer.pad_token_id, text_input_ids, -100
        )
        
        if self.mode == "test":
            mesh_data = item['mesh_data']
            if mesh_data[:7].equal(EDGE):
                mesh_data = mesh_data[7:]
            
            return {
                "text_input_ids": text_input_ids,
                "text_attention_mask": text_attention_mask,
                "text_labels": text_labels,
                "svg_tensors": sample.long(),
                "mesh_data": mesh_data,
            }
            
        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_labels": text_labels,
            "svg_tensors": sample.long(),
        }


class UnderstandingOfflineBasicDataset(Dataset):
    """
    obtrain the data offline
    
    """
    PROMPT_PREFIX = "Please generate few keywords to describe the following SVG:"
    PROMPT_SUFFIX = "#begin:"
    RESPONSE_TEMPLATE = "Here are some keywords: {keywords}"

    def __init__(self, content, tokenizer, mode="train", max_svg_len=1024, max_text_length=64, svg_pad_token_id=0) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.mode = mode
        self.max_text_length = max_text_length
        self.max_svg_len = max_svg_len
        self.content = content
        self.svg_pad_token_id = svg_pad_token_id

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        item = self.content[idx]
        keywords, sample = item['keys'], item['zs']
        response = self.RESPONSE_TEMPLATE.format(keywords=', '.join(keywords))

        prompt_prefix = self.tokenizer(
            self.PROMPT_PREFIX, 
            return_tensors="pt",
        )
        prompt_prefix_ids = prompt_prefix.input_ids[0][:-1]
        prompt_prefix_attention_mask = prompt_prefix.attention_mask[0][:-1]

        prompt_suffix = self.tokenizer(
            self.PROMPT_SUFFIX,
            return_tensors="pt",
        )
        prompt_suffix_ids = prompt_suffix.input_ids[0][1:-1]
        prompt_suffix_attention_mask = prompt_suffix.attention_mask[0][1:-1]
        
        response = self.tokenizer(
            response,
            padding="max_length", 
            truncation=True, 
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        response_ids = response.input_ids[0][1:]
        response_attention_mask = response.attention_mask[0][1:]
        response_labels = torch.where(
            response_ids != self.tokenizer.pad_token_id, response_ids, -100
        )

        sample = sample[:self.max_svg_len]  # prevent too long num path
        sample = pad_tensor(sample, self.max_svg_len, 0, self.svg_pad_token_id)
        
        # create svg_id attention mask
        svg_attention_mask = (sample != self.svg_pad_token_id).to(response_attention_mask.dtype)

        return {
            "prompt_prefix_ids": prompt_prefix_ids,
            "prompt_prefix_attention_mask": prompt_prefix_attention_mask,
            "prompt_suffix_ids": prompt_suffix_ids,
            "prompt_suffix_attention_mask": prompt_suffix_attention_mask,
            "response_ids": response_ids,
            "response_attention_mask": response_attention_mask,
            "response_labels": response_labels,
            "svg_tensors": sample.long(),
            "svg_attention_mask": svg_attention_mask,
        }


class UnderstandingDataCollator:

    def __call__(self, batch):
        """
        args:
            batch - list of (tensor, label)
        """
        prompt_prefix_ids = [x['prompt_prefix_ids'] for x in batch]
        prompt_prefix_attention_mask = [x['prompt_prefix_attention_mask'] for x in batch]
        prompt_suffix_ids = [x['prompt_suffix_ids'] for x in batch]
        prompt_suffix_attention_mask = [x['prompt_suffix_attention_mask'] for x in batch]
        response_ids = [x['response_ids'] for x in batch]
        response_attention_mask = [x['response_attention_mask'] for x in batch]
        response_labels = [x['response_labels'] for x in batch]
        svg_tensors = [x['svg_tensors'] for x in batch]
        svg_attention_mask = [x['svg_attention_mask'] for x in batch]
        
        # pad according to max_len
        svg_tensors = torch.stack(svg_tensors, dim=0).long()
        prompt_prefix_ids = torch.stack(prompt_prefix_ids, dim=0)
        prompt_prefix_attention_mask = torch.stack(prompt_prefix_attention_mask, dim=0)
        prompt_suffix_ids = torch.stack(prompt_suffix_ids, dim=0)
        prompt_suffix_attention_mask = torch.stack(prompt_suffix_attention_mask, dim=0)
        response_ids = torch.stack(response_ids, dim=0)
        response_attention_mask = torch.stack(response_attention_mask, dim=0)
        response_labels = torch.stack(response_labels, dim=0)
        svg_attention_mask = torch.stack(svg_attention_mask, dim=0)

        return {
            "prompt_prefix_ids": prompt_prefix_ids,
            "prompt_prefix_attention_mask": prompt_prefix_attention_mask,
            "prompt_suffix_ids": prompt_suffix_ids,
            "prompt_suffix_attention_mask": prompt_suffix_attention_mask,
            "response_ids": response_ids,
            "response_attention_mask": response_attention_mask,
            "response_labels": response_labels,
            "svg_tensors": svg_tensors, 
            "svg_attention_mask": svg_attention_mask,
        }


class VQDataCollator:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """
    def __init__(self, max_svg_length=512, pad_token_id=0, cluster_batch=False, return_all_token_mask=False, offline_mode=True, mode="train"):
        self.max_svg_length = max_svg_length
        self.pad_token_id = pad_token_id
        self.cluster_batch = cluster_batch
        self.return_all_token_mask = return_all_token_mask
        self.offline_mode = offline_mode
        self.mode = mode

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
        """
        text_input_ids = [x['text_input_ids'] for x in batch]
        text_attention_mask = [x['text_attention_mask'] for x in batch]
        text_labels = [x['text_labels'] for x in batch]
        svg_tensors = [x['svg_tensors'] for x in batch]
        
            
        if self.cluster_batch:
            # find longest sequence
            max_len = max(map(lambda x: x.shape[0], svg_tensors))
            max_len = min(max_len, self.max_svg_length)
        else:
            max_len = self.max_svg_length

        # pad according to max_len
        svg_tensors = list(map(lambda x: pad_tensor(x, max_len, 0, self.pad_token_id), svg_tensors))
        svg_tensors = torch.stack(svg_tensors, dim=0).long()
        text_input_ids = torch.stack(text_input_ids, dim=0)
        text_attention_mask = torch.stack(text_attention_mask, dim=0)
        text_labels = torch.stack(text_labels, dim=0)

        # get padding mask
        if self.return_all_token_mask:
            svg_padding_mask = ~(svg_tensors == self.pad_token_id)
        else:
            svg_padding_mask = ~(svg_tensors == self.pad_token_id).all(dim=2, keepdim=True).squeeze()

        # create padding mask
        if not self.offline_mode:  # only online mode needs it
            svg_padding_mask = list(map(lambda x: cal_compress_padding_mask(x), svg_padding_mask))
            svg_padding_mask = torch.stack(svg_padding_mask, dim=0)

        if self.mode == "test":
            mesh_data = [x['mesh_data'] for x in batch]
            mesh_data = list(map(lambda x: pad_tensor(x, 9012, 0, self.pad_token_id), mesh_data))
            return {
                "input_ids": text_input_ids,
                "attention_mask": text_attention_mask,
                "labels": text_labels,
                "decoder_input_ids": svg_tensors, 
                "decoder_attention_mask": svg_padding_mask,
                "raw_data": torch.stack(mesh_data, dim=0).long(),
            }
        
        return {
            "input_ids": text_input_ids,
            "attention_mask": text_attention_mask,
            "labels": text_labels,
            "decoder_input_ids": svg_tensors, 
            "decoder_attention_mask": svg_padding_mask,
        }

    def __call__(self, batch):
        return self.pad_collate(batch)


class VQSeq2SeqData:
    def __init__(self, config, vq_svg_file, svg_begin_token, tokenizer, offline_mode=True, mode="train", task="generation", inferece_nums=-1, codebook_size=4096, val_data_num=2048, use_custom_collate_fn=False):  
        self.cfg = config
        self.tokenizer = tokenizer
        self.task = task
        self.codebook_size = codebook_size
        content = None
        if mode == "test":
            content = auto_read_data(vq_svg_file)
            if inferece_nums == -1:
                inferece_nums = len(content)
            content = content[:inferece_nums]
            print_c(f"num of testing data: {len(content)}", color='magenta')
            self.pred_data = content
        else:  # for training setting
            if os.path.isdir(vq_svg_file): # read data sequencially
                all_file_path = auto_read_dir(vq_svg_file)
                raw_content = [auto_read_data(item) for item in all_file_path]
                content = [item for sublist in raw_content for item in sublist]
            else: # directly read data from file
                content = auto_read_data(vq_svg_file) ## Load VQSVG data
            num_valid_data = min(int(len(content) * 0.1), val_data_num)
            print_c(f"num of valid data: {num_valid_data}", color='magenta')
            print_c(f"num of train data: {len(content) - num_valid_data}", color='magenta')
            self.valid_data = content[:num_valid_data]
            self.train_data = content[num_valid_data:]
        
        self.svg_begin_token = svg_begin_token
        self.offline_mode = offline_mode
        self.use_custom_collate_fn = use_custom_collate_fn
        
    @property
    def train_dataset(self) -> Dataset:
        if self.offline_mode and self.task == "generation":
            return OfflineBasicDataset(
                content=self.train_data,
                max_path_nums=self.cfg.max_path_nums, 
                tokenizer=self.tokenizer,
                max_text_length=self.cfg.max_text_length,
                mode="train",
            )
        elif self.offline_mode and self.task == "understanding":
            return UnderstandingOfflineBasicDataset(
                content=self.train_data,
                tokenizer=self.tokenizer,
                max_svg_len=self.cfg.max_path_nums,
                max_text_length=self.cfg.max_text_length,
                svg_pad_token_id=0,
            )
        else:
            return BasicDataset(
                content=self.train_data,
                min_path_nums=self.cfg.min_path_nums,
                max_path_nums=self.cfg.max_path_nums, 
                tokenizer=self.tokenizer,
                svg_begin_token = self.svg_begin_token,
                max_text_length=self.cfg.max_text_length,
                mode="train",
                cluster_batch=False
            )

    @property
    def valid_dataset(self) -> Dataset:
        if self.offline_mode and self.task == "generation":
            return OfflineBasicDataset(
                content=self.valid_data,
                max_path_nums=self.cfg.max_path_nums, 
                tokenizer=self.tokenizer,
                max_text_length=self.cfg.max_text_length,
                mode="valid",
            )
        elif self.offline_mode and self.task == "understanding":
            return UnderstandingOfflineBasicDataset(
                content=self.valid_data,
                tokenizer=self.tokenizer,
                max_svg_len=self.cfg.max_path_nums,
                max_text_length=self.cfg.max_text_length,
                svg_pad_token_id=0,
            )
        else:
            return BasicDataset(
                content=self.valid_data,
                min_path_nums=self.cfg.min_path_nums,
                max_path_nums=self.cfg.max_path_nums, 
                tokenizer=self.tokenizer,
                svg_begin_token = self.svg_begin_token,
                max_text_length=self.cfg.max_text_length,
                mode="valid",
                cluster_batch=False
            )

    @property
    def predict_dataset(self) -> Dataset:
        if self.pred_data is None:
            return None
        if self.offline_mode:
            return OfflineBasicDataset(
                content=self.pred_data,
                max_path_nums=self.cfg.max_path_nums, 
                tokenizer=self.tokenizer,
                max_text_length=self.cfg.max_text_length,
                mode="test",
                codebook_size=self.codebook_size,
            )
        else:
            return BasicDataset(
                content=self.pred_data,
                min_path_nums=self.cfg.min_path_nums,
                max_path_nums=self.cfg.max_path_nums, 
                tokenizer=self.tokenizer,
                svg_begin_token = self.svg_begin_token,
                max_text_length=self.cfg.max_text_length,
                mode="test",
                cluster_batch=False,
                saint_check=False,
            )
            
            
    def predict_dataloader(self) -> DataLoader:
        if self.predict_dataset is not None:
            return DataLoader(
                self.predict_dataset, 
                batch_size=self.cfg.predict_batch_size, 
                num_workers=self.cfg.dataloader_num_workers, 
                pin_memory=False, drop_last=False, shuffle=False,
                collate_fn=VQDataCollator(self.cfg.max_path_nums, return_all_token_mask=True, mode='test') if self.use_custom_collate_fn else None
            )
            
            
