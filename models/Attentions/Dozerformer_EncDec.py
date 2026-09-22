import math
import torch
from torch import nn
from einops import rearrange
from models.my_method.Attentions.Transformer_EncDec import Encoder, EncoderLayer
from models.my_method.Attentions.SelfAttention_Family import FullAttention, AttentionLayer
from models.my_method.Attentions.dozer_attention import DozerAttention, DozerAttentionLayer
from models.my_method.build_model_util import DI_embedding, TS_Segment, series_decomp_multi
from math import ceil


class dozerformer_Encoder(nn.Module):
    def __init__(self, configs, mode):
        super().__init__()
        self.patch_size = configs.patch_size if mode == 'Seasonal' else configs.trend_patch_size
        self.in_channel = configs.data_dim

        d_model = configs.embed_dim*configs.patch_size
        d_ff = configs.d_ff*configs.patch_size
        # Embedding是非常重要的问题
        self.encoder_val_embedding = DI_embedding(configs.patch_size, configs.embed_dim, configs.dropout)
        self.encoder_segment = TS_Segment(configs.seq_len, configs.patch_size)
        self.encoder_pos_embed = nn.Parameter(torch.randn(1,
                                                            configs.embed_dim,
                                                            self.encoder_segment.seg_num,
                                                            configs.patch_size,
                                                            self.in_channel
                                                            ))
        self.encoder_pre_norm = nn.LayerNorm(d_model)
        self.encoder_norm = nn.LayerNorm(d_model)
        # Attention
        self.encoder = Encoder(
            [EncoderLayer(
                DozerAttentionLayer(
                    DozerAttention(configs.local_window, configs.stride, configs.rand_rate,
                                    configs.vary_len, self.encoder_segment.seg_num,
                                    False,
                                    attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                    d_model,
                    configs.n_heads),
                d_model=d_model,
                d_ff=d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.encoder_depth)
            ],
            norm_layer=None
        )

    def forward(self, x_enc):
        embeddings = self.encoder_val_embedding(rearrange(x_enc, 'b seq_len ts_d -> b 1 seq_len ts_d'))
        # Segment
        patches = self.encoder_segment(embeddings)
        identity = patches
        # Add pos
        patches = patches + self.encoder_pos_embed

        patches = rearrange(patches, 'b d_model seg_num seg_len ts_d -> (b ts_d) seg_num (seg_len d_model)')
        # PreNorm
        patches = self.encoder_pre_norm(patches)

        encoder_output, attns = self.encoder(patches)
        # PostNorm
        encoder_output = self.encoder_norm(encoder_output)

        # skip connection
        encoder_output = rearrange(encoder_output,
                                   '(b ts_d) seg_num (seg_len d_model) -> b d_model seg_num seg_len ts_d',
                                   seg_len=self.patch_size, ts_d=self.in_channel)
        # encoder_output = self.encoder_segment.concat(encoder_output)

        encoder_output = encoder_output + identity

        return encoder_output


