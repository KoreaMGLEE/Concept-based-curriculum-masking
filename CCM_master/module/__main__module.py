import torch.nn as nn
import torch
import copy
import os
from transformers import *
#from transformers import BertForMaskedLM
from model.act import *

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)

    loss = - targets_prob * student_likelihood
    loss = torch.sum(loss, dim = -1)

    return loss
    #return torch.mean(- targets_prob * student_likelihood, dim = -1).sum()


class SPLLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(SPLLoss, self).__init__(*args, **kwargs)
        #self.threshold = 0.1
        self.growing_factor = 1.3
        #self.v = torch.zeros(n_samples).int()

    def forward(self, input, target, threshold):
        #super_loss = nn.functional.nll_loss(input, target, reduction="none")
        super_loss = nn.functional.cross_entropy(input, target, reduction="none", ignore_idx = self.ignore_index)
        self.threshold = threshold
        v = self.spl_loss(super_loss)

        return (super_loss * v).mean()

    def increase_threshold(self):
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}
BertLayerNorm = torch.nn.LayerNorm

class Last_layer(nn.Module):
    def __init__(self, config, o_dim):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, o_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(o_dim))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.decoder(hidden_states)

        return output

class DiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        #hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


class Disc_only(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_

        self.config = config
        self.disc_head = Last_layer(config, 2)

        self.disc_criterion = nn.CrossEntropyLoss()
        #torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_ids, pp_label, _attention_mask_, mask_, lm_label):

        replaced_mask = input_ids == lm_label

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states=True)
        lm_outputs = self.disc_head(outputs[0])

        lm_label = torch.ones_like(lm_label).cuda()
        lm_label = lm_label.masked_fill(replaced_mask, 0)

        disc_loss = self.disc_criterion(lm_outputs.view(-1, 2), lm_label.view(-1))

        return disc_loss


class RTPP_MLM(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        self.mlm_head = Last_layer(config, 30522)
        self.mlm_criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, pp_label, _attention_mask_, mask_, lm_label):

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states=True)

        lm_outputs = self.mlm_head(outputs[0])
        lm_label = lm_label.masked_fill(mask_, -100)

        mlm_loss = self.mlm_criterion(lm_outputs.view(-1, self.config.vocab_size), lm_label.view(-1))

        return mlm_loss


class RTPP_Model(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        self.pp_head = Last_layer(config, 1)
        self.pp_criterion = nn.BCEWithLogitsLoss()
        #torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_ids, pp_label, _attention_mask_, mask_, lm_label):

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True)
        #Outputs
        outputs = self.pp_head(outputs[0])
        #Get loss
        pp_loss = self.pp_criterion(outputs.squeeze(-1), pp_label)

        return pp_loss


class Soft_Distill_Model(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        self.mlm_head = Last_layer(config, 30522)
        #torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_ids, soft_labels, _attention_mask_, mask_, lm_label, sub_label_position):

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True)
        #Outputs
        logits = outputs[0]
        logits = self.mlm_head(logits)

        distill_loss = 0
        for example_idx in range(logits.shape[0]):
            masked_idx = sub_label_position[example_idx]
            l = soft_cross_entropy(logits[example_idx, masked_idx], soft_labels[example_idx])
            distill_loss += l

        return distill_loss
        
class Soft_Distill_Model_Top(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        self.mlm_head = Last_layer(config, 30522)
        #self.fit_dense = nn.Linear(config.hidden_size, 768)

    def forward(self, input_ids, soft_labels, indices, _attention_mask_, t_hidden = None, t_att = None):

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True)
        #Outputs
        logits = outputs[0]
        logits = self.mlm_head(logits)

        if indices == None:
            distill_loss = soft_cross_entropy(logits, soft_labels)
            distill_loss = distill_loss.mean()
        else:
            new_logits = torch.gather(logits, 2, indices)
            distill_loss = soft_cross_entropy(new_logits, soft_labels)
            distill_loss = distill_loss.mean()

        return distill_loss
    
class Soft_Distill_Model_All(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        self.mlm_head = Last_layer(config, 30522)
        #torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_ids, soft_labels, _attention_mask_):

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True)
        #Outputs
        logits = outputs[0]
        logits = self.mlm_head(logits)

        distill_loss = soft_cross_entropy(logits, soft_labels)
        distill_loss = distill_loss.mean()

        return distill_loss

    def get_logit(self, input_ids, _attention_mask_):
        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True)
        #Outputs
        logits = outputs[0]
        logits = self.mlm_head(logits)

        return logits

class Soft_Distill_Model_Top_Tiny(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        self.mlm_head = Last_layer(config, 30522)
        self.fit_dense = nn.Linear(config.hidden_size, 768)
        #torch.nn.init.xavier_uniform_(self.linear.weight)
        self.mse_loss = nn.MSELoss()

    def forward(self, input_ids, soft_labels, indices, _attention_mask_, t_hidden = None, t_att = None):

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        #Outputs
        logits = outputs[0]
        logits = self.mlm_head(logits)

        new_logits = torch.gather(logits, 2, indices)

        distill_loss = soft_cross_entropy(new_logits, soft_labels)
        distill_loss = distill_loss.mean()

        att_loss = 0. 
        rep_loss = 0.

        #s_logit, s_hidden, s_att = outputs[:3]
        s_hidden, s_att, s_value = outputs[-3:]

        t_layer_num = len(t_att)
        s_layer_num = len(s_att)

        assert t_layer_num % s_layer_num == 0
        
        layers_per_block = int(t_layer_num / s_layer_num)
        new_t_att = [t_att[(i + 1) * layers_per_block - 1] for i in range(s_layer_num)]
        new_t_hidden = [t_hidden[i * layers_per_block] for i in range(s_layer_num + 1)]
        
        #for s, t in zip(s_att, new_t_att):
        #    att_loss += self.mse_loss(s,t)
        
        for s, t in zip(s_hidden, new_t_hidden):
            rep_loss += self.mse_loss(self.fit_dense(s),t)

        #loss = att_loss + rep_loss
        loss = rep_loss

        return distill_loss, loss



class Tiny_BERT_PP(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_

        self.config = config

        self.pp_head = Last_layer(config, 1)
        #self.criterion = nn.CrossEntropyLoss()
        #self.disc_head = Last_layer(config, 2)

        self.mse_loss = nn.MSELoss()
        self.fit_dense = nn.Linear(config.hidden_size, 768)
        #self.fit_dense = nn.Linear(config.hidden_size, 1024)
        self.pp_criterion = nn.BCEWithLogitsLoss(reduction = "none")
        #torch.nn.init.xavier_uniform_(self.linear.weight)
        #self.d_criterion = nn.CrossEntropyLoss(reduce = None)

    def forward(self, input_ids, pp_label, _attention_mask_, mask_, lm_label, t_hidden, t_att):

        att_loss = 0. 
        rep_loss = 0.

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        #s_logit, s_hidden, s_att = outputs[:3]
        s_hidden, s_att, s_value = outputs[-3:]

        t_layer_num = len(t_att)
        s_layer_num = len(s_att)

        assert t_layer_num % s_layer_num == 0
        
        layers_per_block = int(t_layer_num / s_layer_num)
        new_t_att = [t_att[(i + 1) * layers_per_block - 1] for i in range(s_layer_num)]
        new_t_hidden = [t_hidden[i * layers_per_block] for i in range(s_layer_num + 1)]
        '''
        for s, t in zip(s_att, new_t_att):
            att_loss += self.mse_loss(s,t)
        '''
        for s, t in zip(s_hidden, new_t_hidden):
            rep_loss += self.mse_loss(self.fit_dense(s),t)
        
        #loss = att_loss + rep_loss
        loss = rep_loss
        #loss = att_loss

        #Get loss      
        pp_outputs = self.pp_head(outputs[0])
        pp_loss = self.pp_criterion(pp_outputs.squeeze(-1), pp_label)

        '''
        fake_mask = mask_.masked_fill(mask_ == False, True)
        fake_mask = fake_mask.masked_fill(mask_ == True, False)

        pp_loss[fake_mask] = pp_loss[fake_mask] * 2
        '''
        pp_loss = torch.mean(pp_loss)


        return loss, pp_loss

class Tiny_BERT_PP_Att(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_

        self.config = config

        self.pp_head = Last_layer(config, 1)
        #self.criterion = nn.CrossEntropyLoss()
        #self.disc_head = Last_layer(config, 2)

        self.mse_loss = nn.MSELoss()
        self.fit_dense = nn.Linear(config.hidden_size, 768)
        #self.fit_dense = nn.Linear(config.hidden_size, 1024)
        self.pp_criterion = nn.BCEWithLogitsLoss(reduction = "none")
        #torch.nn.init.xavier_uniform_(self.linear.weight)
        #self.d_criterion = nn.CrossEntropyLoss(reduce = None)

    def forward(self, input_ids, pp_label, _attention_mask_, mask_, lm_label, t_hidden, t_att):

        att_loss = 0. 
        rep_loss = 0.

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        #s_logit, s_hidden, s_att = outputs[:3]
        s_hidden, s_att, s_value = outputs[-3:]

        t_layer_num = len(t_att)
        s_layer_num = len(s_att)

        assert t_layer_num % s_layer_num == 0
        
        layers_per_block = int(t_layer_num / s_layer_num)
        new_t_att = [t_att[(i + 1) * layers_per_block - 1] for i in range(s_layer_num)]
        new_t_hidden = [t_hidden[i * layers_per_block] for i in range(s_layer_num + 1)]
        
        for s, t in zip(s_att, new_t_att):
            att_loss += self.mse_loss(s,t)
        
        for s, t in zip(s_hidden, new_t_hidden):
            rep_loss += self.mse_loss(self.fit_dense(s),t)
        
        loss = att_loss + rep_loss
        #loss = rep_loss
        #loss = att_loss

        #Get loss      
        pp_outputs = self.pp_head(outputs[0])
        pp_loss = self.pp_criterion(pp_outputs.squeeze(-1), pp_label)

        pp_loss = torch.mean(pp_loss)

        return loss, pp_loss


class MINILM_PP(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_

        self.config = config

        self.pp_head = Last_layer(config, 1)
        #self.criterion = nn.CrossEntropyLoss()
        #self.disc_head = Last_layer(config, 2)

        #self.mse_loss = nn.MSELoss()
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.kl_loss = nn.KLDivLoss(reduction = "batchmean")

        #self.fit_dense = nn.Linear(config.hidden_size, 768)
        #self.fit_dense = nn.Linear(config.hidden_size, 1024)
        self.pp_criterion = nn.BCEWithLogitsLoss(reduction = "none")
        
        #torch.nn.init.xavier_uniform_(self.linear.weight)
        #self.d_criterion = nn.CrossEntropyLoss(reduce = None)

    def forward(self, input_ids, pp_label, _attention_mask_, mask_, lm_label, t_hidden, t_att, t_value):

        att_loss = 0. 
        rep_loss = 0.

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        #s_logit, s_hidden, s_att = outputs[:3]
        s_hidden, s_att, s_value = outputs[-3:]

        t_value = t_value[-1]
        s_value = s_value[-1]

        t_value_dot = torch.matmul(t_value, t_value.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        s_value_dot = torch.matmul(s_value, s_value.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        t_value_dot = torch.softmax(t_value_dot, dim = -1)
        s_value_dot = torch.softmax(s_value_dot, dim = -1)

        att_loss = self.kl_loss(s_att[-1].log(), t_att[-1])
        value_loss = self.kl_loss(s_value_dot.log(), t_value_dot)

        loss = att_loss + value_loss

        #Get pp loss      
        pp_outputs = self.pp_head(outputs[0])
        pp_loss = self.pp_criterion(pp_outputs.squeeze(-1), pp_label)
        pp_loss = torch.mean(pp_loss)

        return loss, pp_loss

class MINILM_Only(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.kl_loss = nn.KLDivLoss(reduction = "batchmean")
        #sself.mse_loss = nn.MSELoss()


    def forward(self, input_ids, _attention_mask_, t_hidden, t_att, t_value):


        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        #s_logit, s_hidden, s_att = outputs[:3]
        s_hidden, s_att, s_value = outputs[-3:]

        t_value = t_value[-1]
        s_value = s_value[-1]

        #print(s_att[-1].shape, t_att[-1].shape, t_value.shape, s_value.shape)
        
        t_value_dot = torch.matmul(t_value, t_value.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        s_value_dot = torch.matmul(s_value, s_value.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        t_value_dot = torch.softmax(t_value_dot, dim = -1)
        s_value_dot = torch.softmax(s_value_dot, dim = -1)

        s_a = torch.softmax(s_att[-1], dim = -1)
        t_a = torch.softmax(t_att[-1], dim = -1)

        #att_loss = self.kl_loss(s_att[-1].log(), t_att[-1])
        att_loss = self.kl_loss(s_a.log(), t_a)
        value_loss = self.kl_loss(s_value_dot.log(), t_value_dot)

        #att_loss = self.mse_loss(s_att[-1], t_att[-1])
        #value_loss = self.mse_loss(s_value_dot, t_value_dot)

        loss = att_loss + value_loss

        return loss


class ALP_KD(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_

        self.config = config
        self.mse_loss = nn.MSELoss()
        #self.fit_dense = nn.Linear(config.hidden_size, 768)
        #torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_ids, _attention_mask_, t_hidden, t_att):

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        #s_logit, s_hidden, s_att = outputs[:3]
        s_hidden, s_att, s_value = outputs[-3:]

        t_hid = torch.stack(t_hidden, dim = 2)
        st_hid = torch.stack(s_hidden, dim = 2)
        att_score = torch.matmul(st_hid, t_hid.transpose(-1, -2))
        att_score = torch.softmax(att_score, dim = -1)
        weight_t_hidd = torch.matmul(att_score, t_hid)

        loss = self.mse_loss(st_hid, weight_t_hidd)

        return loss

class Hidden_KD(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.mse_loss = nn.MSELoss()
        self.config = config        

    def forward(self, input_ids, _attention_mask_, t_hidden, t_att):

        att_loss = 0. 
        rep_loss = 0.

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        s_hidden, s_att, s_value = outputs[-3:]

        t_layer_num = len(t_att)
        s_layer_num = len(s_att)

        assert t_layer_num % s_layer_num == 0
        
        layers_per_block = int(t_layer_num / s_layer_num)
        #new_t_att = [t_att[(i + 1) * layers_per_block - 1] for i in range(s_layer_num)]
        new_t_hidden = [t_hidden[i * layers_per_block] for i in range(s_layer_num + 1)]
         
        t_hid =[]
        s_hid =[]

        for s, t in zip(s_hidden, new_t_hidden):
            t_hid.append(t)
            s_hid.append(s)
            #rep_loss += self.mse_loss(self.fit_dense(s),t)
            #rep_loss += self.mse_loss(s,t)
        t_hid = torch.stack(t_hid, dim = 2)
        s_hid = torch.stack(s_hid, dim = 2)

        loss = self.mse_loss(s_hid,t_hid)

        #loss = rep_loss

        return loss

class Tiny_BERT_only(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_

        self.config = config
        #self.pp_head = Last_layer(config, 2)
        #self.criterion = nn.CrossEntropyLoss()

        self.mse_loss = nn.MSELoss()
        self.fit_dense = nn.Linear(config.hidden_size, 768)
        #torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_ids, _attention_mask_, t_hidden, t_att):

        att_loss = 0. 
        rep_loss = 0.

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        #s_logit, s_hidden, s_att = outputs[:3]
        s_hidden, s_att, s_value = outputs[-3:]

        t_layer_num = len(t_att)
        s_layer_num = len(s_att)

        assert t_layer_num % s_layer_num == 0
        
        layers_per_block = int(t_layer_num / s_layer_num)
        new_t_att = [t_att[(i + 1) * layers_per_block - 1] for i in range(s_layer_num)]
        new_t_hidden = [t_hidden[i * layers_per_block] for i in range(s_layer_num + 1)]
        
        for s, t in zip(s_att, new_t_att):
            att_loss += self.mse_loss(s,t)
        
        for s, t in zip(s_hidden, new_t_hidden):
            rep_loss += self.mse_loss(self.fit_dense(s),t)

        loss = att_loss + rep_loss
        #loss = rep_loss

        return loss


def lr_scheduler(args_lr, optimizer, step, warmup_step, max_step):
    if step <= warmup_step:
        lr = args_lr * (step / warmup_step)
    else:
        #lr = args_lr * (1-step/max_step)
        lr = args_lr / (max_step - warmup_step) * (max_step - step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer, lr

