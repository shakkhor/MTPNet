"""Writes the files the run visualizer reads for one test run.

results/<setting>/run_info.json  parameters and headline metrics, used for the run list
results/<setting>/report.json    run_info plus predictions and actual values split into
                                 trend + seasonal + local, and error summaries

All values stay on the rescaled scale the model is trained on, like the logged MSE/MAE.
"""
import json
import os
from datetime import datetime

import numpy as np

BAD_THRESHOLD = 1.0   # data is standardized on the train split, so 1.0 = off by one typical swing
CHUNK = 256           # windows decomposed at a time, keeps memory flat on large test sets
COMPS = ('whole', 'trend', 'seasonal', 'local')


def decompose(context, segment, period):
    """Split each forecast window into trend + seasonal + local.

    context: (N, C, V) values right before each window, so the trend has no edge effect at its start
    segment: (N, L, V) the window itself
    trend    centered moving average over one day (period steps), end padded with the last value
    seasonal mean detrended value at the same time of day inside the window
    local    what is left
    """
    half = max(period // 2, 1)
    win = 2 * half + 1
    n, L, V = segment.shape
    if context.shape[1] < half:
        pad = np.repeat(segment[:, :1] if context.shape[1] == 0 else context[:, :1], half - context.shape[1], axis=1)
        context = np.concatenate([pad, context], axis=1)
    x = np.concatenate([context[:, -half:], segment, np.repeat(segment[:, -1:], half, axis=1)], axis=1)
    c = np.cumsum(np.pad(x, ((0, 0), (1, 0), (0, 0))), axis=1)
    trend = (c[:, win:win + L] - c[:, :L]) / win
    detrended = segment - trend
    if period < 2:
        seasonal = np.zeros_like(detrended)
    else:
        onehot = np.eye(period)[np.arange(L) % period]                     # (L, period)
        counts = np.maximum(onehot.sum(axis=0), 1)
        phase_mean = np.einsum('lp,nlv->npv', onehot, detrended) / counts[None, :, None]
        seasonal = np.einsum('lp,npv->nlv', onehot, phase_mean)
    return {'whole': segment, 'trend': trend, 'seasonal': seasonal, 'local': detrended - seasonal}


def _json_params(args):
    params = {}
    for k, v in sorted(vars(args).items()):
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, (str, int, float, bool)) or v is None:
            params[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(i, (str, int, float, bool)) for i in v):
            params[k] = list(v)
    params.pop('device', None)
    # the variate mixer only exists in the MTPNet models
    if not str(args.model).startswith('MTPNet'):
        for k in [k for k in params if k.startswith('variate_')]:
            params.pop(k)
    return params


def _rounded(a):
    return np.round(a.astype(np.float64), 4).tolist()


def write_run_report(preds, trues, test_data, folder_path, setting, args, mse, mae):
    """preds, trues: (n_windows, pred_len, n_vars) in test-loader order (shuffle off)."""
    n, L, C = preds.shape
    in_len = test_data.in_len
    data_x = test_data.data_x[:, -C:]
    names = test_data.columns[-C:]
    dates = test_data.dates
    step = (dates.iloc[1] - dates.iloc[0]).total_seconds()
    period = int(round(86400 / step))       # steps per day
    half = max(period // 2, 1)
    ctx_len = min(half, in_len)

    per_step = {c: np.zeros((C, L)) for c in COMPS}
    per_var = {c: np.zeros(C) for c in COMPS}
    bad = np.zeros(C)
    # forecasts laid end to end, one every pred_len steps, for viewing long ranges
    starts = np.arange(0, n, L)
    stitched_t = {c: np.zeros((C, len(starts) * L)) for c in COMPS}
    stitched_p = {c: np.zeros((C, len(starts) * L)) for c in COMPS}

    for a in range(0, n, CHUNK):
        idx = np.arange(a, min(a + CHUNK, n))
        ctx = np.stack([data_x[i + in_len - ctx_len: i + in_len] for i in idx])
        t = decompose(ctx, trues[idx], period)
        p = decompose(ctx, preds[idx], period)
        for c in COMPS:
            err = np.abs(p[c] - t[c])
            per_step[c] += err.sum(axis=0).T
            per_var[c] += err.sum(axis=(0, 1))
        bad += (np.abs(preds[idx] - trues[idx]) > BAD_THRESHOLD).sum(axis=(0, 1))
        for k in np.nonzero(idx % L == 0)[0]:
            b = idx[k] // L
            for c in COMPS:
                stitched_t[c][:, b * L:(b + 1) * L] = t[c][k].T
                stitched_p[c][:, b * L:(b + 1) * L] = p[c][k].T

    for c in COMPS:
        per_step[c] /= n
        per_var[c] /= n * L
    bad /= n * L

    ms = dates.values.astype('datetime64[ms]').astype(np.int64)
    info = {
        'id': setting,
        'created': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'model': args.model,
        'data': args.data,
        'params': _json_params(args),
        'metrics': {'mse': float(mse), 'mae': float(mae), 'bad': float(bad.mean())},
    }
    report = dict(info)
    report.update({
        'vars': list(names),
        'step_ms': int(step * 1000),
        'period': period,
        'pred_len': L,
        'bad_threshold': BAD_THRESHOLD,
        'start_ms': int(ms[in_len]),                # time of the first forecast step
        'truth': {c: _rounded(stitched_t[c]) for c in COMPS},
        'pred': {c: _rounded(stitched_p[c]) for c in COMPS},
        'per_step': {c: _rounded(per_step[c]) for c in COMPS},
        'per_var': {**{c: _rounded(per_var[c]) for c in COMPS}, 'bad': _rounded(bad)},
    })

    with open(os.path.join(folder_path, 'report.json'), 'w') as f:
        json.dump(report, f, separators=(',', ':'))
    with open(os.path.join(folder_path, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=1)