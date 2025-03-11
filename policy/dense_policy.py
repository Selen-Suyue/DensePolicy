from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce
from transformers import BertModel, BertConfig


# precise DSP
class DensePolicy(nn.Module):
    def __init__(self,
            action_dim,
            Tp, # For predict
            Ta, # For execute Ta <= Tp
            obs_feature_dim,
            binary_expansion=False,
            **kwargs):
        super().__init__()
        
        config = BertConfig(hidden_size=obs_feature_dim, num_attention_heads=8, intermediate_size=obs_feature_dim * 4, num_hidden_layers=4)
        self.Tp = Tp
        self.Ta = Ta
        self.binary_expansion = binary_expansion

        self.action_projection = nn.Linear(obs_feature_dim, action_dim)
        self.cross_attention = BertModel(config)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        self.qpos_projection = nn.Linear(action_dim, obs_feature_dim)
    def process_qpos(self, qpos):
        qpos_masked = random_mask_qpos(qpos)
        qpos_projected = self.qpos_projection(qpos_masked)  
        qpos_projected = qpos_projected.unsqueeze(1)  
        return qpos_projected

    def compute_loss(self, readout, actions, qpos):
        condition_readout = readout.unsqueeze(1)  
        condition_qpos = self.process_qpos(qpos)   
        condition = torch.cat([condition_readout, condition_qpos], dim=1)  
        action_pred = torch.zeros(condition.size(0), 1, condition.size(2), device=condition.device)
       
        while action_pred.shape[1] < actions.shape[1]: 
            if self.binary_expansion:
                action_pred = Bexp(action_pred.transpose(1, 2)).transpose(1, 2)
            else:
                action_pred = self.upsample(action_pred.transpose(1, 2)).transpose(1, 2)
            input_action = torch.cat([action_pred,condition],dim=1)

            attention_output = self.cross_attention(inputs_embeds = input_action).last_hidden_state
            action_pred = attention_output[:,:action_pred.shape[1],:]  
        
        action_pred = action_pred[:,:actions.shape[1],:]
        action_pred = self.action_projection(action_pred)

        loss = F.mse_loss(action_pred, actions, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    
    def predict_action(self,readout,qpos):
        condition_readout = readout.unsqueeze(1)  
        condition_qpos = self.process_qpos(qpos)   
        condition = torch.cat([condition_readout, condition_qpos], dim=1)  
        action_pred = torch.zeros(condition.size(0), 1, condition.size(2), device=condition.device)

        while action_pred.shape[1] < self.Tp: 
            if self.binary_expansion:
                action_pred = Bexp(action_pred.transpose(1, 2)).transpose(1, 2)
            else:
                action_pred = self.upsample(action_pred.transpose(1, 2)).transpose(1, 2)
            input_action = torch.cat([action_pred,condition],dim=1)

            attention_output = self.cross_attention(inputs_embeds = input_action).last_hidden_state
            action_pred = attention_output[:,:action_pred.shape[1],:]

        action_pred = action_pred[:,:self.Tp,:]
        action_pred = self.action_projection(action_pred)

        action_pred = action_pred[:,:self.Ta,:] 
        return action_pred
    
def random_mask_qpos(qpos,mask_ratio=0.3):
        mask = (torch.rand_like(qpos) > mask_ratio).float()  
        qpos_masked = qpos * mask
        return qpos_masked

def Bexp(input_tensor):
     
    batch_size, channels, length = input_tensor.shape
    new_length = length * 2
    output_tensor = torch.zeros(batch_size, channels, new_length, device=input_tensor.device, dtype=input_tensor.dtype)

    # Place original values at even indices
    output_tensor[:, :, ::2] = input_tensor

    # Calculate interpolated values at odd indices
    output_tensor[:, :, 1:-1:2] = (input_tensor[:, :, :-1] + input_tensor[:, :, 1:]) / 2.0
    output_tensor[:,:,-1] = input_tensor[:,:,-1]
    return output_tensor