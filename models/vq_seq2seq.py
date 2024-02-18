"""
Code for VQ-SVG-LLAMA
"""
import sys
import random
import torch  
import torch.nn as nn 
import torch.nn.functional as F
from transformers import T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Stack
import copy
from typing import Any, Mapping, Tuple, List, Optional, Dict, Sequence, Union

def freeze_model(model):
    """
    Freeze the model.
    """
    for param in model.parameters():
        param.requires_grad = False

class VQSVGSeq2SeqModel(T5ForConditionalGeneration):  
    def __init__(self, config, tokenizer=None, vqvae=None, codebook_size=4096):  
        super(VQSVGSeq2SeqModel, self).__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        self.codebook_size = codebook_size + 2  # add one for svg end token
        self.svg_end_token_id = codebook_size
        self.svg_begin_token_id = codebook_size + 1
        self.vqvae = vqvae
        
        # decoder
        self.vqvae_embedding = nn.Embedding(self.codebook_size, config.hidden_size)
        self.vqvae_head = nn.Linear(config.hidden_size, self.codebook_size)

        self.post_init()
        
        if config.frozen_llm: 
            print("Attention! encoder is freezed!")
            self.encoder.requires_grad_ = False # only freeze the encoder
            self.shared.requires_grad_ = False  # freeze the text embedding 

    
    def init_vqvae(self, vqvae):
        self.vqvae = vqvae
        self.vqvae.model.eval()
        for param in self.vqvae.model.parameters():
            param.requires_grad = False

    def init_decoder(self):
        print("Attention! Decoder is inited!")
        decoder_config = copy.deepcopy(self.config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
    

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)
        
        
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None, return_dict=None, encoder_outputs=None, past_key_values=None, output_attentions=None, output_hidden_states=None, use_cache=None, **kwargs): 
        """
            text_input_ids: B x L 
            text_attention_mask: B x L,
            labels: B x L,
            svg_tensors: B x L (x l_bins),  depend on offline or online mode
            svg_padding_mask: B x L,
        """
        if self.config.frozen_llm:  # only calculate svg loss when freezen LLM
            self.encoder.requires_grad_ = False # only freeze the encoder
            self.shared.requires_grad_ = False  # freeze the text embedding 
        
        
        bsz = decoder_input_ids.size(0)
        
        # embedding text
        # text_embeddings = self.shared(text_input_ids)
        
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        hidden_states = encoder_outputs[0]
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        
        # quantizied svg tensors with vqvae
        if self.vqvae is not None: # online mode
            if self.vqvae.model.training: # deepspeed will make vqvae training again
                self.vqvae.model.eval()
                freeze_model(self.vqvae.model)
            svg_token_ids = self.vqvae.model.encode_no_grad(decoder_input_ids, start_level=0, end_level=1)
            svg_token_ids = svg_token_ids[0]  # first compress level
        else:  # offline mode
            svg_token_ids = decoder_input_ids
            if not self.training: # eval mode
                if svg_token_ids[:, 0].sum() == 0: # remove the first text begin token
                    svg_token_ids = svg_token_ids[:, 1:]  
      
        compress_svg_max_length = svg_token_ids.size(1)
        golden_svg_tokens = None
        if self.training:  # training mode
            # add svg end token id
            real_svg_lengths = decoder_attention_mask.sum(dim=1)
            for i in range(bsz):
                cur_padding_pos = min(real_svg_lengths[i], compress_svg_max_length - 1)
                svg_token_ids[i, cur_padding_pos] = self.svg_end_token_id
                decoder_attention_mask[i, cur_padding_pos] = True
            golden_svg_tokens = torch.where(decoder_attention_mask, svg_token_ids, -100).to(svg_token_ids.device).long()
        svg_token_embeddings = self.vqvae_embedding(svg_token_ids) # Encode svg tokens
        
        # decoder_attention_mask = decoder_attention_mask.to(attention_mask.dtype)  # prevent the type error
        # decode svg tokens
        
        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=svg_token_embeddings,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.vqvae_head = self.vqvae_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.vqvae_head.weight.device)
            
        svg_logits = self.vqvae_head(sequence_output)
        
        loss = None

        if golden_svg_tokens is not None:
            # Shift so that tokens < n predict n
            shift_svg_logits = svg_logits[:, :-1, :].contiguous()
            shift_golden_svg_tokens = golden_svg_tokens[:, 1:].contiguous()
            shift_svg_logits = shift_svg_logits.view(-1, self.codebook_size)
            shift_golden_svg_tokens = shift_golden_svg_tokens.view(-1)
            loss = F.cross_entropy(shift_svg_logits, shift_golden_svg_tokens)
        
        if not return_dict:
            if not isinstance(encoder_outputs, tuple):
                encoder_outputs = (encoder_outputs,)
            output = (svg_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=svg_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    
    @property
    def model_device(self):
        return next(self.parameters()).device
    