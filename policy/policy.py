import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from policy.tokenizer import Sparse3DEncoder
from policy.transformer import Transformer
from policy.dense_policy import DensePolicy


class DSP(nn.Module):
    def __init__(
        self, 
        Tp = 16, 
        Ta = 16, 
        input_dim = 6,
        obs_feature_dim = 512, 
        action_dim = 10, 
        hidden_dim = 512,
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        dim_feedforward = 2048, 
        dropout = 0.1,
    ):
        super().__init__()
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DensePolicy(action_dim, 
                                                  Tp, 
                                                  Ta, 
                                                  obs_feature_dim, 
                                                )
        self.readout_embed = nn.Embedding(1, hidden_dim)

    def forward(self, cloud, actions = None, qpos=None, batch_size = 24):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        if actions is not None:
                loss = self.action_decoder.compute_loss(readout, actions, qpos)
                return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout, qpos)
            return action_pred