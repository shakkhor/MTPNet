# MTPNet: Decoder Removal + DozerAttention

Comparison of the model architecture before and after replacing the
Transformer decoder with a `Linear(seq_len, pred_len)` projection and
swapping `FullAttention` for sparse `DozerAttention`, plus a walkthrough
of one training epoch.

## Before: Encoder + Decoder (FullAttention)

Each of the `H_depth` pyramid levels (one per `patch_size`) repeats the same
encoder/decoder block; only one level is drawn below.

```mermaid
flowchart TD
    XENC["x_enc (seq_len)"] --> REVIN1["RevIN norm"]
    REVIN1 --> DECOMP1["series_decomp_multi"]
    DECOMP1 --> SEASENC["seasonal x_enc"]
    DECOMP1 --> TRENDENC["trend x_enc"]

    BATCHY["batch_y"] --> DECINP["dec_inp = [label_len history, zeros(pred_len)]"]
    DECINP --> DECOMP2["series_decomp_multi"]
    DECOMP2 --> SEASDEC["seasonal x_dec"]
    DECOMP2 --> TRENDDEC["trend x_dec"]

    subgraph SeasonalEncoder["Seasonal tsformer_Encoder (per level)"]
        SEASENC --> EMB1["DI_embedding + segment + pos"]
        EMB1 --> ENCATTN["EncoderLayer: FullAttention self-attn"]
        ENCATTN --> ENCOUT["encoder_output[level]"]
    end

    subgraph SeasonalDecoder["Seasonal tsformer_Decoder (per level)"]
        SEASDEC --> EMB2["DI_embedding + segment + pos"]
        EMB2 --> DECSELF["DecoderLayer: FullAttention self-attn"]
        DECSELF --> DECCROSS["FullAttention cross-attn vs encoder_output[level]"]
        DECCROSS --> DECOUT["decoder_output[level]"]
    end

    ENCOUT --> DECCROSS
    DECOUT --> CONCAT1["concat levels + Conv2d output_layer"]
    CONCAT1 --> SLICE1["slice last pred_len"]
    SLICE1 --> SEASPRED["seasonal_predict"]

    TRENDENC -.-> TRENDBRANCH["Trend: same Encoder+Decoder (MTPNet)\nor Linear(seq_len,pred_len) (MTPNet_Linear)"]
    TRENDDEC -.-> TRENDBRANCH
    TRENDBRANCH --> TRENDPRED["trend_predict"]

    SEASPRED --> SUM1["+"]
    TRENDPRED --> SUM1
    SUM1 --> REVIN2["RevIN denorm"]
    REVIN2 --> OUT1["output (pred_len)"]
```

Key point: the decoder's **cross-attention** against the encoder output is
the actual forecasting mechanism — each future query position attends into
the encoder's representation of the past to build its prediction.

## After: Encoder-only (DozerAttention)

```mermaid
flowchart TD
    XENC2["x_enc (seq_len)"] --> REVIN3["RevIN norm"]
    REVIN3 --> DECOMP3["series_decomp_multi"]
    DECOMP3 --> SEASENC2["seasonal x_enc"]
    DECOMP3 --> TRENDENC2["trend x_enc"]

    subgraph SeasonalEncoder2["Seasonal tsformer_Encoder (per level)"]
        SEASENC2 --> EMB3["DI_embedding + segment + pos"]
        EMB3 --> DOZER["EncoderLayer: DozerAttention self-attn\n(sparse local_window + stride mask)"]
        DOZER --> ENCOUT2["encoder_output[level]"]
    end

    ENCOUT2 --> CONCAT2["concat levels + Conv2d output_layer"]
    CONCAT2 --> PROJ1["seasonal_proj: Linear(seq_len, pred_len)"]
    PROJ1 --> SEASPRED2["seasonal_predict"]

    TRENDENC2 -.-> TRENDBRANCH2["Trend: same DozerAttention Encoder + final_proj (MTPNet)\nor Linear(seq_len,pred_len) (MTPNet_Linear)"]
    TRENDBRANCH2 --> TRENDPRED2["trend_predict"]

    SEASPRED2 --> SUM2["+"]
    TRENDPRED2 --> SUM2
    SUM2 --> REVIN4["RevIN denorm"]
    REVIN4 --> OUT2["output (pred_len)"]

    XDEC2["x_dec"] -.unused.-> NOTE["(model no longer takes a decoder path)"]
```

Key point: there is no cross-attention step left. The encoder only ever
sees `x_enc` (the past); forecasting the future is done entirely by the
`Linear(seq_len, pred_len)` projection that replaces the decoder.

## One training epoch (`Exp_Main.train`)

```mermaid
flowchart TD
    START(["for epoch in range(train_epochs)"]) --> TRAINMODE["model.train()"]
    TRAINMODE --> LOOP["for (batch_x, batch_y) in train_loader"]

    LOOP --> ZG["optimizer.zero_grad()"]
    ZG --> POB["process_one_batch:\nbuild dec_inp, outputs = model(batch_x, dec_inp)"]
    POB --> SLICEOUT["slice outputs/batch_y to last pred_len"]
    SLICEOUT --> LOSS["loss = criterion(outputs, batch_y)  [L1 or MSE]"]
    LOSS --> BACK["loss.backward()"]
    BACK --> STEP["optimizer.step()"]
    STEP --> LOOP

    LOOP -->|loader exhausted| TRAINLOSS["train_loss = mean(batch losses)"]
    TRAINLOSS --> VALI["vali_loss = vali(val_loader)  [eval mode, no_grad]"]
    VALI --> TEST["test_loss = vali(test_loader)  [eval mode, no_grad]"]
    TEST --> SCHED["scheduler.step(epoch)  [CosineAnnealingWarmRestarts]"]
    SCHED --> ES["early_stopping(vali_loss, model, path)"]
    ES -->|vali_loss improved| SAVE["save checkpoint.pth"]
    ES -->|no improvement, patience exceeded| STOP(["break: Early stopping"])
    ES -->|no improvement, patience left| START
    SAVE --> START
```

Notes:
- `process_one_batch` (in `utils/tools.py`) builds `dec_inp` by concatenating
  the known `label_len` history from `batch_y` with zeros over the `pred_len`
  horizon — this is fed to the model as `x_dec` regardless of whether the
  model actually consumes it (only the decoder-based architecture does).
- Validation and test both reuse `vali()`, run in `model.eval()` with
  `torch.no_grad()`, so no gradient updates happen there.
- Early stopping tracks `vali_loss` only; `test_loss` is logged per epoch but
  never used to pick the checkpoint.
