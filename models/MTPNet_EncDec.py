import math
import torch
from torch import nn
from einops import rearrange
from models.Attentions.Transformer_EncDec import Encoder, EncoderLayer
from models.Attentions.dozer_attention import DozerAttention, DozerAttentionLayer
from models.build_model_util import DI_embedding, TS_Segment


class tsformer_Encoder(nn.Module):
    def __init__(self, configs, mode):
        super().__init__()
        self.patch_size = configs.patch_size if mode == 'Seasonal' else configs.trend_patch_size
        self.in_channel = configs.data_dim

        self.H_depth = len(self.patch_size)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_val_embeddings = nn.ModuleList()
        self.encoder_segments = nn.ModuleList()
        self.encoder_pos_embeds = nn.ParameterList()
        self.encoder_pre_norms = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_feat_fuse = nn.ModuleList()
        for i in range(self.H_depth):
            H_patch_size = self.patch_size[i]
            d_model_lvl = configs.embed_dim * H_patch_size
            dff_lvl = configs.d_ff * H_patch_size

            self.encoder_segments.append(TS_Segment(configs.seq_len, H_patch_size))

            self.encoder_blocks.append(
                Encoder(
                    [EncoderLayer(
                        DozerAttentionLayer(
                            DozerAttention(configs.local_window, configs.stride, configs.rand_rate,
                                           configs.vary_len, self.encoder_segments[i].seg_num,
                                           False,
                                           attention_dropout=configs.dropout,
                                           output_attention=configs.output_attention),
                            d_model_lvl,
                            configs.n_heads),
                        d_model=d_model_lvl,
                        d_ff=dff_lvl,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.encoder_depth)
                    ],
                    norm_layer=None
                ))

            self.encoder_val_embeddings.append(DI_embedding(H_patch_size, configs.embed_dim, configs.dropout))
            self.encoder_pos_embeds.append(nn.Parameter(torch.randn(1,
                                                                    configs.embed_dim,
                                                                    self.encoder_segments[i].seg_num,
                                                                    H_patch_size,
                                                                    self.in_channel
                                                                    )))
            self.encoder_pre_norms.append(nn.LayerNorm(d_model_lvl))
            self.encoder_norms.append(nn.LayerNorm(d_model_lvl))
            if i > 0:
                self.encoder_feat_fuse.append(nn.Conv2d(in_channels=2*configs.embed_dim,
                                                out_channels=configs.embed_dim,
                                                kernel_size=(1, 1)))

    def forward(self, x_enc):
        encoder_outputs = []
        for i in range(self.H_depth):
            # value embedding
            patches = self.encoder_val_embeddings[i](rearrange(x_enc, 'b seq_len ts_d -> b 1 seq_len ts_d'))
            identity = patches

            if i > 0:
                encoder_output_pre_layer = self.encoder_segments[i - 1].concat(encoder_outputs[i - 1])
                patches = self.encoder_feat_fuse[i - 1](torch.cat((patches, encoder_output_pre_layer), 1))
            # Segment
            patches = self.encoder_segments[i](patches)
            # Add pos
            patches = patches + self.encoder_pos_embeds[i]

            patches = rearrange(patches, 'b d_model seg_num seg_len ts_d -> (b ts_d) seg_num (seg_len d_model)')
            # PreNorm
            patches = self.encoder_pre_norms[i](patches)

            encoder_output, attns = self.encoder_blocks[i](patches)
            # PostNorm
            encoder_output = self.encoder_norms[i](encoder_output)

            # skip connection
            encoder_output = rearrange(encoder_output,
                                       '(b ts_d) seg_num (seg_len d_model) -> b d_model seg_num seg_len ts_d',
                                       seg_len=self.patch_size[i], ts_d=self.in_channel)
            encoder_output = self.encoder_segments[i].concat(encoder_output)
            encoder_output = encoder_output + identity
            encoder_output = self.encoder_segments[i](encoder_output)

            encoder_outputs.append(encoder_output)
        return encoder_outputs
