import torch.nn as nn
import torch
import copy
import os
from transformers import *
#from transformers import BertForMaskedLM
from model.act import *

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}
BertLayerNorm = torch.nn.LayerNorm

class Last_layer(nn.Module):
    def __init__(self, config, o_dim):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.transform_act_fn = ACT2FN[config.hidden_act]
        #self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, o_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(o_dim))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        #hidden_states = self.LayerNorm(hidden_states)
        output = self.decoder(hidden_states)

        return output

class ELECTRA(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_

        self.config = config
        self.disc_head = Last_layer(config, 2)
        #self.disc_head = DiscriminatorPredictions(config)

        self.disc_criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, _attention_mask_, mask_, lm_label, t_hidden, t_att):

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True)
        #s_logit, s_hidden, s_att = outputs[:3]
        
        #disc loss
        disc_outputs = self.disc_head(outputs[0])
        #replaced mask
        replaced_mask = input_ids == lm_label
        disc_label = torch.ones_like(lm_label).cuda()
        disc_label = disc_label.masked_fill(replaced_mask, 0)
        #disc_label = pp_label.to(torch.long)
        #Get loss
        disc_loss = self.disc_criterion(disc_outputs.view(-1, 2), disc_label.view(-1))
        
        return disc_loss

class ELECTRA_Distill(nn.Module):
    def __init__(self, base_model_, config, teacher_model_freeze = True):
        super().__init__()
        self.base_model = base_model_

        self.teacher_model_freeze = teacher_model_freeze

        self.config = config
        self.disc_head = Last_layer(config, 2)

        self.mse_loss = nn.MSELoss()
        #self.fit_dense = nn.Linear(config.hidden_size // 4, 256)
        self.fit_dense = nn.Linear(config.hidden_size, config.hidden_size // 4)
        self.disc_criterion = nn.CrossEntropyLoss()
        #torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_ids, _attention_mask_, mask_, lm_label, t_hidden, t_att):

        att_loss = 0. 
        rep_loss = 0.

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True)
        #s_logit, s_hidden, s_att = outputs[:3]
        s_hidden, s_att = outputs[-2:]

        t_layer_num = len(t_att)
        s_layer_num = len(s_att)

        assert t_layer_num % s_layer_num == 0
        
        layers_per_block = int(t_layer_num / s_layer_num)
        if self.teacher_model_freeze:
            new_t_att = [t_att[(i + 1) * layers_per_block - 1].detach() for i in range(s_layer_num)]
            new_t_hidden = [t_hidden[i * layers_per_block].detach() for i in range(s_layer_num + 1)]
        else:
            new_t_att = [t_att[(i + 1) * layers_per_block - 1] for i in range(s_layer_num)]
            new_t_hidden = [t_hidden[i * layers_per_block] for i in range(s_layer_num + 1)]
        '''
        for s, t in zip(s_att, new_t_att):
            att_loss += self.mse_loss(s,t)
        '''
        for s, t in zip(s_hidden, new_t_hidden):
            rep_loss += self.mse_loss(self.fit_dense(s), t)
            #rep_loss += self.mse_loss(self.fit_dense(s), t)

        #loss = att_loss + rep_loss
        loss = rep_loss
        
        #disc loss
        disc_outputs = self.disc_head(outputs[0])
        #replaced mask
        replaced_mask = input_ids == lm_label
        disc_label = torch.ones_like(lm_label).cuda()
        disc_label = disc_label.masked_fill(replaced_mask, 0)
        #disc_label = pp_label.to(torch.long)
        #Get loss
        disc_loss = self.disc_criterion(disc_outputs.view(-1, 2), disc_label.view(-1))
        
        return loss, disc_loss