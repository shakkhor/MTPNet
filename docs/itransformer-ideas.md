# Ideas from iTransformer for MTPNet

Notes on what [iTransformer: Inverted Transformers Are Effective for Time
Series Forecasting](https://arxiv.org/abs/2310.06625) (Liu et al., ICLR 2024)
suggests for this model, ranked by expected value vs. effort.

## Paper summary

iTransformer "inverts" the Transformer: instead of one token per time step,
**each variate's whole lookback series is one token** (`Linear(T -> D)`).

- **Attention** runs over the variate axis (an `N x N` map), capturing
  multivariate correlations.
- **FFN**, shared across variates, learns temporal patterns per token.
- **LayerNorm** normalizes each variate token.
- **Projection** is an MLP from each token to `pred_len`.
- No positional embedding; time features (`x_mark`) are embedded as extra
  variate tokens.

Main results (lookback 96, MSE averaged over horizons 96/192/336/720):

| Dataset | iTransformer | PatchTST |
|---|---|---|
| ECL | **0.178** | 0.205 |
| Traffic | **0.428** | 0.481 |
| Solar | **0.233** | 0.270 |
| PEMS | **0.119** | 0.217 |
| ETT (avg) | 0.383 | **0.381** |

Other findings:

- **Ablation**: attention along the time axis instead of across variates
  hurts badly on Traffic (0.913 vs 0.428).
- **Lookback**: iTransformer keeps improving as the lookback grows;
  vanilla Transformers plateau or degrade.
- **Partial-variate training**: training on a random 20% of variates per
  batch still generalizes to all variates, with much lower memory.
- **Limitation**: no gain on low-dimensional data such as ETT (7 variates).

## Gap in MTPNet

MTPNet is **fully channel-independent**. In `tsformer_Encoder.forward`
(`models/MTPNet_EncDec.py`), patches are reshaped to
`(b ts_d) seg_num (seg_len d_model)`, so DozerAttention only attends over
time within a single variate. No component lets variates exchange
information. That is exactly what iTransformer adds.

## Ideas

### 1. Variate-attention stage before `final_proj`

Cheapest and most direct test of the paper's main claim.

In `MTPNet.forward` (`models/MTPNet.py`), right before `self.final_proj`,
`final_predict` has shape `[b, ts_d, seq_len]`, which is already
iTransformer's token layout (one token per variate, dim `seq_len`).
Insert 1-2 inverted encoder layers there (attention over `ts_d`, then an
FFN), with a residual connection so the model can't do worse than the
current one.

- About 20 lines; leaves the multi-scale temporal encoder untouched.
- Results in a two-stage model: temporal (existing), then cross-variate
  (new).
- Put it behind a flag, e.g. `--variate_attn`.

### 2. Parallel iTransformer branch

Embed the RevIN-normalized input with `Linear(seq_len -> D)`, run inverted
layers, project to `pred_len`, and add the result to the MTPNet output
(like the existing seasonal + trend sum).

- More parameters than idea 1.
- Clean ablation: MTPNet vs. iTransformer vs. MTPNet + iTransformer.

### 3. Timestamp features as extra tokens

iTransformer embeds time features (`x_mark`) as extra variate tokens.
MTPNet currently ignores them completely: `process_one_batch`
(`utils/tools.py`) never passes marks to the model.

- Nearly free once idea 1 or 2 exists.

### 4. Scale the lookback

MTPNet's head is a linear `seq_len -> pred_len` map, the kind of design the
paper shows benefits from long lookbacks.

- Sweep `seq_len` in {96, 336, 720}.
- Re-check `patch_size` / `seg_num` and Dozer `stride` / `local_window`,
  which are tuned for 96.

### 5. Partial-variate training (Traffic/ECL scale only)

Currently blocked because `encoder_pos_embeds` is per-channel (shape
`(1, embed_dim, seg_num, patch_size, in_channel)`) and RevIN has
per-channel affine parameters.

- Share the positional embedding across channels to enable it; this also
  acts as a regularizer.
- Then sample a random subset of variates per batch during training.

### 6. MLP projection head

Replace `final_proj` (a single `nn.Linear`) with Linear-GELU-Linear, as in
iTransformer's projection.

- Small expected gain; one-line experiment.

## Evaluation caveat

The `run.py` defaults are ETTh1 (`data_dim=7`). The paper found variate
attention **does not help on ETT**, so ideas 1-3 will likely look flat
there. Evaluate them on **ECL, Traffic, Solar, or Weather**, where
cross-variate structure matters. If ETT is the main benchmark, ideas 4 and
6 are the more promising ones.

## Suggested order

1. Idea 1 on ECL or Weather (small, easy to switch off, tests the main
   claim).
2. Idea 4 lookback sweep (no code changes).
3. Idea 3 if idea 1 helps.
4. Ideas 2, 5, 6 as follow-ups.