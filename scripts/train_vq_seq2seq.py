import transformers
import types
import yaml
from dataclasses import dataclass, field
from transformers import Trainer
from models.vq_seq2seq import VQSVGSeq2SeqModel
from data.vqseq2seq_dataset import VQDataCollator, VQSeq2SeqData
from typing import Optional


@dataclass
class VQVAEConfig:
    config_path: str = field(default=None)
    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    vq_svg_pad_file: str = field(default=None, metadata={"help": "Path to the vq svg pad file."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    freezen_llm: bool = field(default=False)
    init_decoder: bool = field(default=False)
    

def load_yaml_config(config_path):  
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration as a dictionary.

    """
    
    def dict_to_simplenamespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = dict_to_simplenamespace(value)
            return types.SimpleNamespace(**d)
        elif isinstance(d, list):
            return [dict_to_simplenamespace(item) for item in d]
        else:
            return d
    
    print("load config files from {}".format(config_path))
    with open(config_path, 'r') as config_file:  
        try:  
            config = yaml.safe_load(config_file)  
            config = dict_to_simplenamespace(config)
        except yaml.YAMLError as exc:  
            print(exc)  
            return None  
    print("config loaded successfully!")
    print("config: {}".format(config), "green", "underline")
    print()
    return config


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class CustomTrainier(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, **kwargs):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            **kwargs,
        )
 
    def compute_loss(self, model, inputs, return_outputs=False):
        
        outputs = model(**inputs)
        loss = None
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss 
    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, VQVAEConfig))
    model_args, data_args, training_args, vqvae_args = parser.parse_args_into_dataclasses()
    
    # parsing vqvae_config:
    vqvae_config = load_yaml_config(vqvae_args.config_path)

    # config 
    flant5config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    flant5config.frozen_llm = training_args.freezen_llm
    flant5config.max_text_length = 64
    flant5config.min_path_nums = 4
    flant5config.max_path_nums = 512
    flant5config.use_cache = False

    flant5_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,  # 512
        padding_side="right",
        use_fast=True,
    )
    
    svg_data_module = VQSeq2SeqData(
        flant5config, 
        data_args.data_path, 
        tokenizer=flant5_tokenizer, 
        offline_mode=True,
        task="generation",
        svg_begin_token = None,
        codebook_size = vqvae_config.vqvae.l_bins,
    )

    data_collator = VQDataCollator(
        max_svg_length=flant5config.max_path_nums,
        offline_mode=True,
        return_all_token_mask=True, # for offline setting
    )
    
    data_module = dict(
        train_dataset=svg_data_module.train_dataset, 
        eval_dataset=svg_data_module.valid_dataset, 
        data_collator=data_collator
    )

    SvgSeq2SeqModel = VQSVGSeq2SeqModel.from_pretrained(
        model_args.model_name_or_path, 
        config=flant5config,
        codebook_size=vqvae_config.vqvae.l_bins,
        cache_dir=training_args.cache_dir,
        tokenizer=flant5_tokenizer,
    )

    if training_args.init_decoder:
        SvgSeq2SeqModel.init_decoder()

    SvgSeq2SeqModel.is_parallelizable = False
    SvgSeq2SeqModel.model_parallel = False

    trainer = CustomTrainier(model=SvgSeq2SeqModel, tokenizer=flant5_tokenizer, args=training_args, **data_module)

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()