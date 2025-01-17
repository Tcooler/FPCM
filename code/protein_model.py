from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import esm
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead
from protein_modules import TransformerLayer,MLP
import random

class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
        min_layernumber=28
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout
        self.min_layernumber = min_layernumber
        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )
        layers_transformer = []
        for i in range(self.num_layers):
            if i < self.min_layernumber:
                with_lora = False
            else:
                with_lora = True 
            
            layers_transformer.append(
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                    with_lora = with_lora,
                )
            )
        self.layers = nn.ModuleList(layers_transformer)
            

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]


class Model(nn.Module):
    def __init__(self,min_layernumber=28,class_category=2,input_protein_dim=1280,last_hidden_dim=1280,output_dim=512,attention_heads=20,protein_num_layers=33,num_mlp_layer=2,activation='relu'):
        super(Model,self).__init__()
        self.constant = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        self.esm = ESM2(min_layernumber=min_layernumber,embed_dim=input_protein_dim,attention_heads=attention_heads,num_layers=protein_num_layers)
        self.batch_converter = self.esm.alphabet.get_batch_converter()
        self.last_hidden_dim = last_hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.protein_num_layers = protein_num_layers
        self.mask_category = len(self.constant)
        self.class_category = class_category
        self.normal_layer = nn.LayerNorm(1280)
        self.class_classify = MLP(1280,[256] * (num_mlp_layer - 1) + [self.class_category],activation=self.activation)
    
    def mask_x(self,x):
        pred_positions,mlm_Y = [],[]
        x_mlm = list(x)
        for i, token in enumerate(x_mlm):
            if random.random() < 0.15:
                masked_token = None
                if random.random() < 0.8:
                    masked_token = '<mask>'
                else:
                    if random.random() < 0.5:
                        masked_token = token
                    else:
                        masked_token = self.constant[random.randint(0,self.mask_category-1)]
                t = torch.zeros(self.mask_category,dtype = torch.float)
                t[self.constant.index(token)] = 1.0
                mlm_Y.append(t)
                pred_positions.append(i)
                x_mlm[i] = masked_token
        return ''.join(x_mlm),pred_positions,mlm_Y

    
    def forward(self,labels, tokens,device,mask_flag=False):
        input = list(zip(labels,tokens))
        batch_labels, batch_strs, tokens_now = self.batch_converter(input)
        if mask_flag:
            tokens_new = []
            for i,x in enumerate(tokens):
                x_tmp,_,_ = self.mask_x(x)
                tokens_new.append(x_tmp)
            inputdata = list(zip(labels,tokens_new))
            batch_labels, batch_strs, tokens_new = self.batch_converter(inputdata)
            tokens_now = torch.cat([tokens_now,tokens_new],0)
        tokens_now = tokens_now.to(device)
        batch_lens = (tokens_now != self.esm.alphabet.padding_idx).sum(1)
        tokens_now = self.esm(tokens_now, repr_layers=[self.protein_num_layers])
        token_representations = tokens_now["representations"][self.protein_num_layers]
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 0])
            #sequence_representations.append(torch.mean(token_representations[i, 1:tokens_len],dim=0))
        sequence_representations = torch.stack(sequence_representations,dim=0)
        class_output = self.normal_layer(sequence_representations)
        class_output = self.class_classify(class_output)
        return class_output,sequence_representations
