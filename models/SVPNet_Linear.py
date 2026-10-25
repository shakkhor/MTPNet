import torch
from torch import nn
from einops import rearrange

from models.SVPNet_EncDec import SVPNet_Encoder
from models.Attentions.REVIN import RevIN
from models.build_model_util import series_decomp_multi, VariateMixer


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mode = configs.mode
        assert self.mode in ["pretrain", 'finetune', "forecasting"], "Error mode."
        self.patch_size = configs.patch_size
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.in_channel = configs.data_dim
        self.embed_dim = configs.embed_dim
        self.encoder_depth = configs.encoder_depth
        self.dropout = configs.dropout
        self.epoch = 0
        configs.activation = 'gelu'

        self.revin_layer = RevIN(self.in_channel, affine=True, subtract_last=False)
        self.revin_layer_dec = RevIN(self.in_channel, affine=True, subtract_last=False)

        # Decomposition
        self.decomp_multi = series_decomp_multi(configs.moving_avg)

        # Seasonal encoder and decoder
        self.encoder_seasonal = SVPNet_Encoder(configs, mode='Seasonal')
        self.output_layer = nn.Conv2d(in_channels=self.embed_dim * self.encoder_seasonal.H_depth,
                                      out_channels=1,
                                      kernel_size=(1, 1))

        # Trend Encoder
        self.trend_model = nn.Linear(configs.seq_len, configs.pred_len)

        # Cross-variate attention on the seasonal branch (iTransformer-style)
        # Tokens are one variate at one scale, length embed_dim*seq_len
        self.variate_mixer = VariateMixer(self.embed_dim * configs.seq_len, configs.variate_d_model, configs.n_heads,
                                          configs.variate_d_ff, configs.variate_layers,
                                          configs.dropout)

        # Seasonal projection: seq_len -> pred_len
        self.seasonal_proj = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
                ) -> torch.tensor:
        x_enc = self.revin_layer(x_enc, 'norm')
        x_enc, trend_enc = self.decomp_multi(x_enc)

        # Seasonal
        encoder_output = self.encoder_seasonal(x_enc)


        # one token per variate per scale, scales stacked on the variate axis
        # example 32 8 8 12 7 -> 32 7 8*8*12 scale 1
        # example 32 8 4 24 7 -> 32 7 8*4*24 scale 2
        # combine two scales as the input for transformer 32 14 8*8*12
        # concat drops the padding of patch sizes that don't divide seq_len, so every scale has length seq_len
        segments = self.encoder_seasonal.encoder_segments
        final_predict = rearrange(segments[0].concat(encoder_output[0]), 'b d seq_len ts_d -> b ts_d (d seq_len)')
        for i in range(1, self.encoder_seasonal.H_depth):
            final_predict = torch.cat((final_predict, rearrange(segments[i].concat(encoder_output[i]), 'b d seq_len ts_d -> b ts_d (d seq_len)')), dim=1)
        # apply variate mixer here for multi scale
        final_predict = self.variate_mixer(final_predict)
        # back to scales stacked on the channel axis: 32 14 8*96 -> 32 2*8 96 7
        final_predict = rearrange(final_predict, 'b (s ts_d) (d seq_len) -> b (s d) seq_len ts_d',
                                  ts_d=self.in_channel, d=self.embed_dim)
        final_predict = self.output_layer(final_predict)

        # Trend
        trend_enc = rearrange(trend_enc, 'b seq_len ts_d -> b ts_d seq_len')
        trend_predict = self.trend_model(trend_enc)
        trend_predict = rearrange(trend_predict, 'b ts_d seq_len -> b seq_len ts_d')

        # Concate Trend and Seasonal
        final_predict = rearrange(final_predict, 'b 1 seq_len ts_d -> b ts_d seq_len')
        final_predict = self.seasonal_proj(final_predict)
        final_predict = rearrange(final_predict, 'b ts_d pred_len -> b pred_len ts_d')
        final_predict += trend_predict

        final_predict = self.revin_layer(final_predict, 'denorm')

        return final_predict

